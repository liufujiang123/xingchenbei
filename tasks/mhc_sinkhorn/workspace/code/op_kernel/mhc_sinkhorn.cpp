// AIV-only correctness baseline: one core owns complete N x N matrices.
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

template <class DT_LOGITS, uint64_t N, uint64_t MASK_MODE>
class KernelMhcSinkhorn {
public:
    __aicore__ inline KernelMhcSinkhorn() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights,
                                const MhcSinkhornTilingData &tilingData) {
        matrixCount = tilingData.matrixCount;
        matrixSize = tilingData.matrixSize;
        usedCoreNum = tilingData.usedCoreNum;
        iterations = tilingData.iterations;
        eps = tilingData.eps;

        logitsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(logits),
                                 matrixCount * matrixSize);
        weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(weights),
                                  matrixCount * matrixSize);
        if constexpr (MASK_MODE == 1) {
            maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(mask), 1);
        } else if constexpr (MASK_MODE == 2) {
            maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(mask),
                                   matrixCount * matrixSize);
        }

        constexpr uint32_t storageBytes = N == 8
            ? Align32(MATRIX_BATCH * N * N * sizeof(DT_LOGITS))
            : MATRIX_BATCH * N * 32;
        constexpr uint32_t computeBytes = MATRIX_BATCH * 8 * 8 * sizeof(float);
        pipe.InitBuffer(inputQueue, 1, storageBytes);
        pipe.InitBuffer(outputQueue, 1, storageBytes);
        pipe.InitBuffer(stateBuffer, computeBytes);
        pipe.InitBuffer(rowStatsBuffer, MATRIX_BATCH * 32);
        pipe.InitBuffer(rowBroadcastBuffer, computeBytes);
        if constexpr (MASK_MODE == 2 && AscendC::IsSameType<DT_LOGITS, half>::value) {
            pipe.InitBuffer(maskFloatBuffer, computeBytes);
        }
    }

    __aicore__ inline void Process() {
        const uint64_t coreId = AscendC::GetBlockIdx();
        const uint64_t base = matrixCount / usedCoreNum;
        const uint64_t extra = matrixCount % usedCoreNum;
        const uint64_t localCount = base + (coreId < extra ? 1 : 0);
        const uint64_t firstMatrix = coreId * base + (coreId < extra ? coreId : extra);
        for (uint64_t localIndex = 0; localIndex < localCount; localIndex += MATRIX_BATCH) {
            const uint32_t batchCount = static_cast<uint32_t>(
                localCount - localIndex < MATRIX_BATCH ? localCount - localIndex : MATRIX_BATCH);
            if constexpr (N == 8) {
                ProcessBatchN8(firstMatrix + localIndex, batchCount);
            } else {
                ProcessBatchPadded(firstMatrix + localIndex, batchCount);
            }
        }
    }

private:
    static constexpr uint32_t MATRIX_BATCH = N == 8 ? 32 : 4;

    __aicore__ static constexpr uint32_t Align32(uint32_t bytes) {
        return (bytes + 31U) / 32U * 32U;
    }

    __aicore__ inline void SyncVectorToScalar() {
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
    }

    __aicore__ inline void SyncScalarToVector() {
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
    }

    __aicore__ inline void LoadStorage(AscendC::LocalTensor<DT_LOGITS> local,
                                       const AscendC::GlobalTensor<DT_LOGITS> &global,
                                       uint64_t offset, uint32_t elementCount) {
        const uint32_t copyBytes = elementCount * sizeof(DT_LOGITS);
        AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_LOGITS> padParams{
            false, 0, 0, static_cast<DT_LOGITS>(0)};
        AscendC::DataCopyPad(local, global[offset], copyParams, padParams);
    }

    __aicore__ inline void ProcessBatchN8(uint64_t firstMatrix, uint32_t batchCount) {
        constexpr uint32_t matrixElements = 64;
        const uint32_t elementCount = batchCount * matrixElements;
        const uint64_t matrixOffset = firstMatrix * matrixSize;
        AscendC::LocalTensor<float> state = stateBuffer.Get<float>();

        AscendC::LocalTensor<DT_LOGITS> input = inputQueue.AllocTensor<DT_LOGITS>();
        LoadStorage(input, logitsGm, matrixOffset, elementCount);
        inputQueue.EnQue(input);
        input = inputQueue.DeQue<DT_LOGITS>();
        if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
            AscendC::Cast(state, input, AscendC::RoundMode::CAST_NONE, elementCount);
        } else {
            AscendC::Adds(state, input, 0.0f, elementCount);
        }
        inputQueue.FreeTensor(input);

        if constexpr (MASK_MODE == 1) {
            input = inputQueue.AllocTensor<DT_LOGITS>();
            LoadStorage(input, maskGm, 0, 1);
            inputQueue.EnQue(input);
            input = inputQueue.DeQue<DT_LOGITS>();
            const float maskScalar = static_cast<float>(input.GetValue(0));
            inputQueue.FreeTensor(input);
            AscendC::Adds(state, state, maskScalar, elementCount);
        } else if constexpr (MASK_MODE == 2) {
            input = inputQueue.AllocTensor<DT_LOGITS>();
            LoadStorage(input, maskGm, matrixOffset, elementCount);
            inputQueue.EnQue(input);
            input = inputQueue.DeQue<DT_LOGITS>();
            if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
                AscendC::LocalTensor<float> maskFloat = maskFloatBuffer.Get<float>();
                AscendC::Cast(maskFloat, input, AscendC::RoundMode::CAST_NONE, elementCount);
                AscendC::Add(state, state, maskFloat, elementCount);
            } else {
                AscendC::Add(state, state, input, elementCount);
            }
            inputQueue.FreeTensor(input);
        }

        StableRowSoftmaxN8(state, batchCount);
        NormalizeColumnsN8(state, batchCount);
        for (uint32_t iteration = 1; iteration < iterations; ++iteration) {
            NormalizeRowsN8(state, batchCount);
            NormalizeColumnsN8(state, batchCount);
        }

        AscendC::LocalTensor<DT_LOGITS> output = outputQueue.AllocTensor<DT_LOGITS>();
        if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
            AscendC::Cast(output, state, AscendC::RoundMode::CAST_NONE, elementCount);
        } else {
            AscendC::Adds(output, state, 0.0f, elementCount);
        }
        outputQueue.EnQue(output);
        output = outputQueue.DeQue<DT_LOGITS>();
        AscendC::DataCopyExtParams copyParams{
            1, elementCount * static_cast<uint32_t>(sizeof(DT_LOGITS)), 0, 0, 0};
        AscendC::DataCopyPad(weightsGm[matrixOffset], output, copyParams);
        outputQueue.FreeTensor(output);
    }

    __aicore__ inline void ProcessBatchPadded(uint64_t firstMatrix, uint32_t batchCount) {
        constexpr uint32_t storageRowElements = 32 / sizeof(DT_LOGITS);
        const uint64_t matrixOffset = firstMatrix * matrixSize;
        AscendC::LocalTensor<float> state = stateBuffer.Get<float>();
        AscendC::Duplicate(state, 0.0f, batchCount * 64);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<DT_LOGITS> input = inputQueue.AllocTensor<DT_LOGITS>();
        for (uint32_t matrix = 0; matrix < batchCount; ++matrix) {
            for (uint32_t row = 0; row < N; ++row) {
                LoadStorage(input[(matrix * N + row) * storageRowElements], logitsGm,
                            matrixOffset + matrix * matrixSize + row * N, N);
            }
        }
        inputQueue.EnQue(input);
        input = inputQueue.DeQue<DT_LOGITS>();
        for (uint32_t matrix = 0; matrix < batchCount; ++matrix) {
            for (uint32_t row = 0; row < N; ++row) {
                if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
                    AscendC::Cast(state[matrix * 64 + row * 8],
                                  input[(matrix * N + row) * storageRowElements],
                                  AscendC::RoundMode::CAST_NONE, N);
                } else {
                    AscendC::Adds(state[matrix * 64 + row * 8],
                                  input[(matrix * N + row) * storageRowElements], 0.0f, N);
                }
            }
        }
        inputQueue.FreeTensor(input);

        if constexpr (MASK_MODE == 1) {
            input = inputQueue.AllocTensor<DT_LOGITS>();
            LoadStorage(input, maskGm, 0, 1);
            inputQueue.EnQue(input);
            input = inputQueue.DeQue<DT_LOGITS>();
            const float maskScalar = static_cast<float>(input.GetValue(0));
            inputQueue.FreeTensor(input);
            const AscendC::UnaryRepeatParams rowParams(1, 1, 1, 1);
            AscendC::Adds(state, state, maskScalar, N,
                          static_cast<uint8_t>(batchCount * 8), rowParams);
        } else if constexpr (MASK_MODE == 2) {
            input = inputQueue.AllocTensor<DT_LOGITS>();
            for (uint32_t matrix = 0; matrix < batchCount; ++matrix) {
                for (uint32_t row = 0; row < N; ++row) {
                    LoadStorage(input[(matrix * N + row) * storageRowElements], maskGm,
                                matrixOffset + matrix * matrixSize + row * N, N);
                }
            }
            inputQueue.EnQue(input);
            input = inputQueue.DeQue<DT_LOGITS>();
            if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
                AscendC::LocalTensor<float> maskFloat = maskFloatBuffer.Get<float>();
                AscendC::Duplicate(maskFloat, 0.0f, batchCount * 64);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t matrix = 0; matrix < batchCount; ++matrix) {
                    for (uint32_t row = 0; row < N; ++row) {
                        AscendC::Cast(maskFloat[matrix * 64 + row * 8],
                                      input[(matrix * N + row) * storageRowElements],
                                      AscendC::RoundMode::CAST_NONE, N);
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
                const AscendC::BinaryRepeatParams rowParams(1, 1, 1, 1, 1, 1);
                AscendC::Add(state, state, maskFloat, N,
                             static_cast<uint8_t>(batchCount * 8), rowParams);
            } else {
                for (uint32_t matrix = 0; matrix < batchCount; ++matrix) {
                    for (uint32_t row = 0; row < N; ++row) {
                        AscendC::Add(state[matrix * 64 + row * 8],
                                     state[matrix * 64 + row * 8],
                                     input[(matrix * N + row) * storageRowElements], N);
                    }
                }
            }
            inputQueue.FreeTensor(input);
        }

        StableRowSoftmaxPadded(state, batchCount);
        NormalizeColumnsPadded(state, batchCount);
        for (uint32_t iteration = 1; iteration < iterations; ++iteration) {
            NormalizeRowsPadded(state, batchCount);
            NormalizeColumnsPadded(state, batchCount);
        }

        AscendC::LocalTensor<DT_LOGITS> output = outputQueue.AllocTensor<DT_LOGITS>();
        for (uint32_t matrix = 0; matrix < batchCount; ++matrix) {
            for (uint32_t row = 0; row < N; ++row) {
                if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
                    AscendC::Cast(output[(matrix * N + row) * storageRowElements],
                                  state[matrix * 64 + row * 8],
                                  AscendC::RoundMode::CAST_NONE, N);
                } else {
                    AscendC::Adds(output[(matrix * N + row) * storageRowElements],
                                  state[matrix * 64 + row * 8], 0.0f, N);
                }
            }
        }
        outputQueue.EnQue(output);
        output = outputQueue.DeQue<DT_LOGITS>();
        for (uint32_t matrix = 0; matrix < batchCount; ++matrix) {
            for (uint32_t row = 0; row < N; ++row) {
                AscendC::DataCopyExtParams copyParams{
                    1, N * static_cast<uint32_t>(sizeof(DT_LOGITS)), 0, 0, 0};
                AscendC::DataCopyPad(
                    weightsGm[matrixOffset + matrix * matrixSize + row * N],
                    output[(matrix * N + row) * storageRowElements], copyParams);
            }
        }
        outputQueue.FreeTensor(output);
    }

    __aicore__ inline void ProcessMatrix(uint64_t matrixIndex) {
        constexpr uint32_t elementCount = N * N;
        const uint64_t matrixOffset = matrixIndex * matrixSize;
        AscendC::LocalTensor<float> state = stateBuffer.Get<float>();

        AscendC::LocalTensor<DT_LOGITS> input = inputQueue.AllocTensor<DT_LOGITS>();
        if constexpr (N == 8) {
            LoadStorage(input, logitsGm, matrixOffset, elementCount);
        } else {
            constexpr uint32_t storageRowElements = 32 / sizeof(DT_LOGITS);
            for (uint32_t row = 0; row < N; ++row) {
                LoadStorage(input[row * storageRowElements], logitsGm,
                            matrixOffset + row * N, N);
            }
        }
        inputQueue.EnQue(input);
        input = inputQueue.DeQue<DT_LOGITS>();
        if constexpr (N == 8) {
            if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
                AscendC::Cast(state, input, AscendC::RoundMode::CAST_NONE, elementCount);
            } else {
                AscendC::Adds(state, input, 0.0f, elementCount);
            }
        } else {
            constexpr uint32_t storageRowElements = 32 / sizeof(DT_LOGITS);
            for (uint32_t row = 0; row < N; ++row) {
                if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
                    AscendC::Cast(state[row * 8], input[row * storageRowElements],
                                  AscendC::RoundMode::CAST_NONE, N);
                } else {
                    AscendC::Adds(state[row * 8], input[row * storageRowElements], 0.0f, N);
                }
            }
        }
        inputQueue.FreeTensor(input);

        if constexpr (MASK_MODE == 1) {
            input = inputQueue.AllocTensor<DT_LOGITS>();
            LoadStorage(input, maskGm, 0, 1);
            inputQueue.EnQue(input);
            input = inputQueue.DeQue<DT_LOGITS>();
            const float maskScalar = static_cast<float>(input.GetValue(0));
            inputQueue.FreeTensor(input);
            if constexpr (N == 8) {
                AscendC::Adds(state, state, maskScalar, elementCount);
            } else {
                for (uint32_t row = 0; row < N; ++row) {
                    AscendC::Adds(state[row * 8], state[row * 8], maskScalar, N);
                }
            }
        } else if constexpr (MASK_MODE == 2) {
            input = inputQueue.AllocTensor<DT_LOGITS>();
            if constexpr (N == 8) {
                LoadStorage(input, maskGm, matrixOffset, elementCount);
            } else {
                constexpr uint32_t storageRowElements = 32 / sizeof(DT_LOGITS);
                for (uint32_t row = 0; row < N; ++row) {
                    LoadStorage(input[row * storageRowElements], maskGm,
                                matrixOffset + row * N, N);
                }
            }
            inputQueue.EnQue(input);
            input = inputQueue.DeQue<DT_LOGITS>();
            if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
                AscendC::LocalTensor<float> maskFloat = maskFloatBuffer.Get<float>();
                if constexpr (N == 8) {
                    AscendC::Cast(maskFloat, input, AscendC::RoundMode::CAST_NONE, elementCount);
                    AscendC::Add(state, state, maskFloat, elementCount);
                } else {
                    constexpr uint32_t storageRowElements = 32 / sizeof(DT_LOGITS);
                    for (uint32_t row = 0; row < N; ++row) {
                        AscendC::Cast(maskFloat[row * 8], input[row * storageRowElements],
                                      AscendC::RoundMode::CAST_NONE, N);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t row = 0; row < N; ++row) {
                        AscendC::Add(state[row * 8], state[row * 8], maskFloat[row * 8], N);
                    }
                }
            } else {
                if constexpr (N == 8) {
                    AscendC::Add(state, state, input, elementCount);
                } else {
                    constexpr uint32_t storageRowElements = 32 / sizeof(DT_LOGITS);
                    for (uint32_t row = 0; row < N; ++row) {
                        AscendC::Add(state[row * 8], state[row * 8],
                                     input[row * storageRowElements], N);
                    }
                }
            }
            inputQueue.FreeTensor(input);
        }

        if constexpr (N == 8) {
            StableRowSoftmaxN8(state);
            NormalizeColumnsN8(state);
            for (uint32_t iteration = 1; iteration < iterations; ++iteration) {
                NormalizeRowsN8(state);
                NormalizeColumnsN8(state);
            }
        } else {
            StableRowSoftmaxPadded(state);
            NormalizeColumnsPadded(state);
            for (uint32_t iteration = 1; iteration < iterations; ++iteration) {
                NormalizeRowsPadded(state);
                NormalizeColumnsPadded(state);
            }
        }

        AscendC::LocalTensor<DT_LOGITS> output = outputQueue.AllocTensor<DT_LOGITS>();
        if constexpr (N == 8) {
            if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
                AscendC::Cast(output, state, AscendC::RoundMode::CAST_NONE, elementCount);
            } else {
                AscendC::Adds(output, state, 0.0f, elementCount);
            }
        } else {
            constexpr uint32_t storageRowElements = 32 / sizeof(DT_LOGITS);
            for (uint32_t row = 0; row < N; ++row) {
                if constexpr (AscendC::IsSameType<DT_LOGITS, half>::value) {
                    AscendC::Cast(output[row * storageRowElements], state[row * 8],
                                  AscendC::RoundMode::CAST_NONE, N);
                } else {
                    AscendC::Adds(output[row * storageRowElements], state[row * 8], 0.0f, N);
                }
            }
        }
        outputQueue.EnQue(output);
        output = outputQueue.DeQue<DT_LOGITS>();
        if constexpr (N == 8) {
            AscendC::DataCopyExtParams copyParams{
                1, elementCount * static_cast<uint32_t>(sizeof(DT_LOGITS)), 0, 0, 0};
            AscendC::DataCopyPad(weightsGm[matrixOffset], output, copyParams);
        } else {
            constexpr uint32_t storageRowElements = 32 / sizeof(DT_LOGITS);
            for (uint32_t row = 0; row < N; ++row) {
                AscendC::DataCopyExtParams copyParams{
                    1, N * static_cast<uint32_t>(sizeof(DT_LOGITS)), 0, 0, 0};
                AscendC::DataCopyPad(weightsGm[matrixOffset + row * N],
                                     output[row * storageRowElements], copyParams);
            }
        }
        outputQueue.FreeTensor(output);
    }

    __aicore__ inline void StableRowSoftmaxPadded(const AscendC::LocalTensor<float> &state,
                                                  uint32_t batchCount = 1) {
        AscendC::LocalTensor<float> rowStats = rowStatsBuffer.Get<float>();
        AscendC::LocalTensor<float> broadcast = rowBroadcastBuffer.Get<float>();
        const AscendC::BinaryRepeatParams rowBinaryParams(1, 1, 1, 1, 1, 1);
        const AscendC::UnaryRepeatParams rowUnaryParams(1, 1, 1, 1);
        AscendC::Duplicate(rowStats, 0.0f, batchCount * 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax<float, true>(
            rowStats, state, N, static_cast<uint8_t>(batchCount * 8), 1, 1, 1,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastRowsN8(rowStats, broadcast, batchCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(state, state, broadcast, N, static_cast<uint8_t>(batchCount * 8),
                     rowBinaryParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(state, state, N, static_cast<uint8_t>(batchCount * 8), rowUnaryParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, true>(
            rowStats, state, static_cast<uint8_t>(batchCount), 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastRowsN8(rowStats, broadcast, batchCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(state, state, broadcast, N, static_cast<uint8_t>(batchCount * 8),
                     rowBinaryParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(state, state, eps, N, static_cast<uint8_t>(batchCount * 8),
                      rowUnaryParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalizeRowsPadded(const AscendC::LocalTensor<float> &state,
                                               uint32_t batchCount = 1) {
        AscendC::LocalTensor<float> rowStats = rowStatsBuffer.Get<float>();
        AscendC::LocalTensor<float> broadcast = rowBroadcastBuffer.Get<float>();
        const AscendC::BinaryRepeatParams rowBinaryParams(1, 1, 1, 1, 1, 1);
        AscendC::BlockReduceSum<float, true>(
            rowStats, state, static_cast<uint8_t>(batchCount), 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(rowStats, rowStats, eps, batchCount * 8);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastRowsN8(rowStats, broadcast, batchCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(state, state, broadcast, N, static_cast<uint8_t>(batchCount * 8),
                     rowBinaryParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalizeColumnsPadded(const AscendC::LocalTensor<float> &state,
                                                  uint32_t batchCount = 1) {
        AscendC::LocalTensor<float> columnStats = rowStatsBuffer.Get<float>();
        const AscendC::UnaryRepeatParams columnInitParams(1, 1, 1, 8);
        const AscendC::BinaryRepeatParams columnAddParams(1, 1, 1, 1, 1, 8);
        const AscendC::BinaryRepeatParams columnBroadcastParams(1, 1, 0, 1, 1, 0);
        AscendC::Adds(columnStats, state, eps, N, static_cast<uint8_t>(batchCount),
                      columnInitParams);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 1; row < N; ++row) {
            AscendC::Add(columnStats, columnStats, state[row * 8], N,
                         static_cast<uint8_t>(batchCount), columnAddParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t matrix = 0; matrix < batchCount; ++matrix) {
            AscendC::Div(state[matrix * 64], state[matrix * 64], columnStats[matrix * 8],
                         N, N, columnBroadcastParams);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BroadcastRowsN8(const AscendC::LocalTensor<float> &rowStats,
                                           const AscendC::LocalTensor<float> &broadcast,
                                           uint32_t batchCount = 1) {
        AscendC::Brcb(broadcast, rowStats, static_cast<uint8_t>(batchCount), {1, 8});
    }

    __aicore__ inline void StableRowSoftmaxN8(const AscendC::LocalTensor<float> &state,
                                              uint32_t batchCount = 1) {
        AscendC::LocalTensor<float> rowStats = rowStatsBuffer.Get<float>();
        AscendC::LocalTensor<float> broadcast = rowBroadcastBuffer.Get<float>();
        AscendC::BlockReduceMax<float, true>(
            rowStats, state, static_cast<uint8_t>(batchCount), 64, 1, 1, 8);
        BroadcastRowsN8(rowStats, broadcast, batchCount);
        AscendC::Sub(state, state, broadcast, batchCount * 64);
        AscendC::Exp(state, state, batchCount * 64);
        AscendC::BlockReduceSum<float, true>(
            rowStats, state, static_cast<uint8_t>(batchCount), 64, 1, 1, 8);
        BroadcastRowsN8(rowStats, broadcast, batchCount);
        AscendC::Div(state, state, broadcast, batchCount * 64);
        AscendC::Adds(state, state, eps, batchCount * 64);
    }

    __aicore__ inline void NormalizeRowsN8(const AscendC::LocalTensor<float> &state,
                                           uint32_t batchCount = 1) {
        AscendC::LocalTensor<float> rowStats = rowStatsBuffer.Get<float>();
        AscendC::LocalTensor<float> broadcast = rowBroadcastBuffer.Get<float>();
        AscendC::BlockReduceSum<float, true>(
            rowStats, state, static_cast<uint8_t>(batchCount), 64, 1, 1, 8);
        AscendC::Adds(rowStats, rowStats, eps, batchCount * 8);
        BroadcastRowsN8(rowStats, broadcast, batchCount);
        AscendC::Div(state, state, broadcast, batchCount * 64);
    }

    __aicore__ inline void NormalizeColumnsN8(const AscendC::LocalTensor<float> &state,
                                              uint32_t batchCount = 1) {
        AscendC::LocalTensor<float> columnStats = rowStatsBuffer.Get<float>();
        AscendC::LocalTensor<float> pairSums = rowBroadcastBuffer.Get<float>();
        AscendC::LocalTensor<float> quadSums = pairSums[MATRIX_BATCH * 32];
        const AscendC::BinaryRepeatParams pairParams(1, 1, 1, 1, 2, 2);
        const AscendC::UnaryRepeatParams pairEpsParams(1, 1, 4, 4);
        const AscendC::BinaryRepeatParams treeParams(1, 1, 1, 1, 2, 2);
        AscendC::Add(pairSums, state, state[8], 8,
                     static_cast<uint8_t>(batchCount * 4), pairParams);
        AscendC::Adds(pairSums, pairSums, eps, 8, static_cast<uint8_t>(batchCount),
                      pairEpsParams);
        AscendC::Add(quadSums, pairSums, pairSums[8], 8,
                     static_cast<uint8_t>(batchCount * 2), treeParams);
        AscendC::Add(columnStats, quadSums, quadSums[8], 8,
                     static_cast<uint8_t>(batchCount), treeParams);
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 1;
        repeatParams.src0RepStride = 1;
        repeatParams.src1RepStride = 0;
        for (uint32_t matrix = 0; matrix < batchCount; ++matrix) {
            AscendC::Div(state[matrix * 64], state[matrix * 64], columnStats[matrix * 8],
                         8, 8, repeatParams);
        }
    }

    __aicore__ inline void StableRowSoftmax(const AscendC::LocalTensor<float> &state) {
        SyncVectorToScalar();
        for (uint32_t row = 0; row < N; ++row) {
            const uint32_t rowOffset = row * N;
            float maximum = state.GetValue(rowOffset);
            for (uint32_t column = 1; column < N; ++column) {
                const float value = state.GetValue(rowOffset + column);
                maximum = value > maximum ? value : maximum;
            }
            for (uint32_t column = 0; column < N; ++column) {
                const uint32_t offset = rowOffset + column;
                state.SetValue(offset, state.GetValue(offset) - maximum);
            }
        }
        SyncScalarToVector();
        AscendC::Exp(state, state, N * N);
        SyncVectorToScalar();
        for (uint32_t row = 0; row < N; ++row) {
            const uint32_t rowOffset = row * N;
            float rowSum = 0.0f;
            for (uint32_t column = 0; column < N; ++column) {
                rowSum += state.GetValue(rowOffset + column);
            }
            const float reciprocal = 1.0f / rowSum;
            for (uint32_t column = 0; column < N; ++column) {
                const uint32_t offset = rowOffset + column;
                state.SetValue(offset, state.GetValue(offset) * reciprocal + eps);
            }
        }
    }

    __aicore__ inline void NormalizeRows(const AscendC::LocalTensor<float> &state) {
        for (uint32_t row = 0; row < N; ++row) {
            const uint32_t rowOffset = row * N;
            float rowSum = eps;
            for (uint32_t column = 0; column < N; ++column) {
                rowSum += state.GetValue(rowOffset + column);
            }
            const float reciprocal = 1.0f / rowSum;
            for (uint32_t column = 0; column < N; ++column) {
                const uint32_t offset = rowOffset + column;
                state.SetValue(offset, state.GetValue(offset) * reciprocal);
            }
        }
    }

    __aicore__ inline void NormalizeColumns(const AscendC::LocalTensor<float> &state) {
        for (uint32_t column = 0; column < N; ++column) {
            float columnSum = eps;
            for (uint32_t row = 0; row < N; ++row) {
                columnSum += state.GetValue(row * N + column);
            }
            const float reciprocal = 1.0f / columnSum;
            for (uint32_t row = 0; row < N; ++row) {
                const uint32_t offset = row * N + column;
                state.SetValue(offset, state.GetValue(offset) * reciprocal);
            }
        }
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowStatsBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowBroadcastBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskFloatBuffer;
    AscendC::GlobalTensor<DT_LOGITS> logitsGm;
    AscendC::GlobalTensor<DT_LOGITS> maskGm;
    AscendC::GlobalTensor<DT_LOGITS> weightsGm;
    uint64_t matrixCount = 0;
    uint64_t matrixSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t iterations = 0;
    float eps = 0.0f;
};

template <typename DT_LOGITS, uint64_t N, uint64_t MASK_MODE>
__global__ __aicore__ void mhc_sinkhorn(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights,
                                        GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcSinkhornTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tilingData, tiling);
    KernelMhcSinkhorn<DT_LOGITS, N, MASK_MODE> op;
    op.Init(logits, mask, weights, tilingData);
    op.Process();
}
