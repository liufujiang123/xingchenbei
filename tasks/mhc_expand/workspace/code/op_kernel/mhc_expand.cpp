// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_expand_tiling.h"
#include "tiling_key_mhc_expand.h"

template <class DT_X, uint64_t MHC_MULT_KIND>
class KernelMhcExpand {
public:
    __aicore__ inline KernelMhcExpand() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR o, const MhcExpandTilingData &tilingData) {
        s = tilingData.s;
        d = tilingData.d;
        mhcMult = tilingData.mhcMult;
        tileLength = tilingData.tileLength;
        usedCoreNum = tilingData.usedCoreNum;
        mode = tilingData.mode;

        const uint64_t sd = s * d;
        const uint64_t expandedSize = sd * mhcMult;
        const uint64_t inputSize = mode == 0 ? sd : expandedSize;
        const uint64_t outputSize = mode == 0 ? expandedSize : sd;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), inputSize);
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o), outputSize);

        const uint32_t publicTileBytes = tileLength * sizeof(DT_X);
        if (mode == 0) {
            pipe.InitBuffer(forwardQueue, 1, publicTileBytes);
        } else {
            const uint64_t reductionExtent = GetReductionExtent();
            if (reductionExtent > 1) {
                pipe.InitBuffer(inputQueue, 2, publicTileBytes);
            } else {
                pipe.InitBuffer(inputQueue, 1, publicTileBytes);
            }
            pipe.InitBuffer(outputQueue, 1, publicTileBytes);
            if constexpr (AscendC::IsSameType<DT_X, half>::value) {
                if (reductionExtent != 2) {
                    pipe.InitBuffer(convertedBuf, tileLength * sizeof(float));
                    pipe.InitBuffer(accumulatorBuf, tileLength * sizeof(float));
                }
            } else {
                pipe.InitBuffer(convertedBuf, tileLength * sizeof(float));
                pipe.InitBuffer(accumulatorBuf, tileLength * sizeof(float));
            }
        }
    }
    __aicore__ inline void Process() {
        const uint64_t tilesPerRow = d / tileLength + (d % tileLength == 0 ? 0 : 1);
        const uint64_t taskCount = s * tilesPerRow;
        const uint64_t coreId = AscendC::GetBlockIdx();
        const uint64_t base = taskCount / usedCoreNum;
        const uint64_t extra = taskCount % usedCoreNum;
        const uint64_t coreTaskCount = base + (coreId < extra ? 1 : 0);
        const uint64_t coreTaskStart = coreId * base + (coreId < extra ? coreId : extra);

        if constexpr (MHC_MULT_KIND != 0) {
            ProcessBackward(coreTaskStart, coreTaskCount, tilesPerRow);
        } else if (mode == 0) {
            ProcessForward(coreTaskStart, coreTaskCount, tilesPerRow);
        } else {
            ProcessBackward(coreTaskStart, coreTaskCount, tilesPerRow);
        }
    }

private:
    __aicore__ inline uint64_t GetReductionExtent() const {
        if constexpr (MHC_MULT_KIND != 0) {
            return MHC_MULT_KIND;
        }
        return mhcMult;
    }

    __aicore__ inline uint32_t GetValidLength(uint64_t dStart) const {
        const uint64_t remaining = d - dStart;
        return static_cast<uint32_t>(remaining < tileLength ? remaining : tileLength);
    }

    __aicore__ inline void ProcessForward(uint64_t taskStart, uint64_t taskCount,
                                          uint64_t tilesPerRow) {
        const uint64_t alignedRowLength = ((d + 15) / 16) * 16;
        const uint64_t maxBatchRows = alignedRowLength == 0 ? 0 : tileLength / alignedRowLength;
        const uint64_t outputRowGapBytes = (mhcMult - 1) * d * sizeof(DT_X);
        if (tilesPerRow == 1 && maxBatchRows > 1 &&
            outputRowGapBytes <= static_cast<uint64_t>(UINT32_MAX)) {
            ProcessForwardRows(taskStart, taskCount, maxBatchRows,
                               static_cast<uint32_t>(outputRowGapBytes));
            return;
        }

        for (uint64_t localTask = 0; localTask < taskCount; ++localTask) {
            const uint64_t taskId = taskStart + localTask;
            const uint64_t sIndex = taskId / tilesPerRow;
            const uint64_t tileInRow = taskId % tilesPerRow;
            const uint64_t dStart = tileInRow * tileLength;
            const uint32_t validLength = GetValidLength(dStart);
            const uint32_t copyBytes = validLength * sizeof(DT_X);
            const uint64_t inputOffset = sIndex * d + dStart;

            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{
                false, 0, 0, static_cast<DT_X>(0)};
            AscendC::LocalTensor<DT_X> xLocal = forwardQueue.AllocTensor<DT_X>();
            AscendC::DataCopyPad(xLocal, xGm[inputOffset], copyParams, padParams);
            forwardQueue.EnQue(xLocal);

            xLocal = forwardQueue.DeQue<DT_X>();
            for (uint64_t k = 0; k < mhcMult; ++k) {
                const uint64_t outputOffset = (sIndex * mhcMult + k) * d + dStart;
                AscendC::DataCopyPad(oGm[outputOffset], xLocal, copyParams);
            }
            forwardQueue.FreeTensor(xLocal);
        }
    }

    __aicore__ inline void ProcessForwardRows(uint64_t rowStart, uint64_t rowCount,
                                              uint64_t maxBatchRows,
                                              uint32_t outputRowGapBytes) {
        constexpr uint64_t MAX_BLOCK_COUNT = 4095;
        const uint64_t batchCapacity = maxBatchRows < MAX_BLOCK_COUNT ? maxBatchRows : MAX_BLOCK_COUNT;
        const uint32_t rowBytes = static_cast<uint32_t>(d * sizeof(DT_X));
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            false, 0, 0, static_cast<DT_X>(0)};

        uint64_t processedRows = 0;
        while (processedRows < rowCount) {
            const uint64_t remainingRows = rowCount - processedRows;
            const uint64_t currentRows64 =
                remainingRows < batchCapacity ? remainingRows : batchCapacity;
            const uint16_t currentRows = static_cast<uint16_t>(currentRows64);
            const uint64_t currentRowStart = rowStart + processedRows;
            const uint64_t inputOffset = currentRowStart * d;

            AscendC::DataCopyExtParams inputParams{currentRows, rowBytes, 0, 0, 0};
            AscendC::LocalTensor<DT_X> xLocal = forwardQueue.AllocTensor<DT_X>();
            AscendC::DataCopyPad(xLocal, xGm[inputOffset], inputParams, padParams);
            forwardQueue.EnQue(xLocal);

            xLocal = forwardQueue.DeQue<DT_X>();

            AscendC::DataCopyExtParams outputParams{
                currentRows, rowBytes, 0, outputRowGapBytes, 0};
            for (uint64_t k = 0; k < mhcMult; ++k) {
                const uint64_t outputOffset = (currentRowStart * mhcMult + k) * d;
                AscendC::DataCopyPad(oGm[outputOffset], xLocal, outputParams);
            }
            forwardQueue.FreeTensor(xLocal);
            processedRows += currentRows64;
        }
    }

    __aicore__ inline void ProcessBackward(uint64_t taskStart, uint64_t taskCount,
                                           uint64_t tilesPerRow) {
        const uint64_t reductionExtent = GetReductionExtent();
        const uint64_t alignedRowLength = ((d + 15) / 16) * 16;
        const uint64_t maxBatchRows = alignedRowLength == 0 ? 0 : tileLength / alignedRowLength;
        const uint64_t inputRowGapBytes = (reductionExtent - 1) * d * sizeof(DT_X);
        if constexpr (AscendC::IsSameType<DT_X, half>::value) {
            if (reductionExtent == 2) {
                ProcessBackwardHalfM2(taskStart, taskCount, tilesPerRow,
                                      alignedRowLength, maxBatchRows,
                                      inputRowGapBytes);
                return;
            }
        }
        if (tilesPerRow == 1 && maxBatchRows > 1 &&
            inputRowGapBytes <= static_cast<uint64_t>(UINT32_MAX)) {
            ProcessBackwardRows(taskStart, taskCount, alignedRowLength, maxBatchRows,
                                static_cast<uint32_t>(inputRowGapBytes));
            return;
        }

        AscendC::LocalTensor<float> converted = convertedBuf.Get<float>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuf.Get<float>();

        for (uint64_t localTask = 0; localTask < taskCount; ++localTask) {
            const uint64_t taskId = taskStart + localTask;
            const uint64_t sIndex = taskId / tilesPerRow;
            const uint64_t tileInRow = taskId % tilesPerRow;
            const uint64_t dStart = tileInRow * tileLength;
            const uint32_t validLength = GetValidLength(dStart);
            const uint32_t copyBytes = validLength * sizeof(DT_X);
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{
                false, 0, 0, static_cast<DT_X>(0)};
            if (reductionExtent == 1) {
                AscendC::Duplicate(accumulator, 0.0f, validLength);
                const uint64_t inputOffset = sIndex * mhcMult * d + dStart;
                AscendC::LocalTensor<DT_X> xLocal = inputQueue.AllocTensor<DT_X>();
                AscendC::DataCopyPad(xLocal, xGm[inputOffset], copyParams, padParams);
                inputQueue.EnQue(xLocal);

                xLocal = inputQueue.DeQue<DT_X>();
                AscendC::Cast(converted, xLocal, AscendC::RoundMode::CAST_NONE, validLength);
                AscendC::Add(accumulator, accumulator, converted, validLength);
                inputQueue.FreeTensor(xLocal);
            } else {
                const uint64_t firstInputOffset = sIndex * reductionExtent * d + dStart;
                AscendC::LocalTensor<DT_X> firstLocal = inputQueue.AllocTensor<DT_X>();
                AscendC::DataCopyPad(firstLocal, xGm[firstInputOffset], copyParams, padParams);
                inputQueue.EnQue(firstLocal);

                for (uint64_t k = 0; k < reductionExtent; ++k) {
                    AscendC::LocalTensor<DT_X> xLocal = inputQueue.DeQue<DT_X>();
                    if (k + 1 < reductionExtent) {
                        const uint64_t nextInputOffset =
                            (sIndex * reductionExtent + k + 1) * d + dStart;
                        AscendC::LocalTensor<DT_X> nextLocal =
                            inputQueue.AllocTensor<DT_X>();
                        AscendC::DataCopyPad(nextLocal, xGm[nextInputOffset], copyParams,
                                             padParams);
                        inputQueue.EnQue(nextLocal);
                    }
                    if (k == 0) {
                        AscendC::Cast(accumulator, xLocal, AscendC::RoundMode::CAST_NONE,
                                      validLength);
                    } else {
                        AscendC::Cast(converted, xLocal, AscendC::RoundMode::CAST_NONE,
                                      validLength);
                        AscendC::Add(accumulator, accumulator, converted, validLength);
                    }
                    inputQueue.FreeTensor(xLocal);
                }
            }

            AscendC::LocalTensor<DT_X> oLocal = outputQueue.AllocTensor<DT_X>();
            AscendC::Cast(oLocal, accumulator, AscendC::RoundMode::CAST_RINT, validLength);
            outputQueue.EnQue(oLocal);
            oLocal = outputQueue.DeQue<DT_X>();
            const uint64_t outputOffset = sIndex * d + dStart;
            AscendC::DataCopyPad(oGm[outputOffset], oLocal, copyParams);
            outputQueue.FreeTensor(oLocal);
        }
    }

    __aicore__ inline void ProcessBackwardHalfM2(uint64_t taskStart,
                                                 uint64_t taskCount,
                                                 uint64_t tilesPerRow,
                                                 uint64_t alignedRowLength,
                                                 uint64_t maxBatchRows,
                                                 uint64_t inputRowGapBytes) {
        if (tilesPerRow == 1 && maxBatchRows > 1 &&
            inputRowGapBytes <= static_cast<uint64_t>(UINT32_MAX)) {
            ProcessBackwardHalfM2Rows(taskStart, taskCount, alignedRowLength,
                                      maxBatchRows,
                                      static_cast<uint32_t>(inputRowGapBytes));
            return;
        }

        for (uint64_t localTask = 0; localTask < taskCount; ++localTask) {
            const uint64_t taskId = taskStart + localTask;
            const uint64_t sIndex = taskId / tilesPerRow;
            const uint64_t tileInRow = taskId % tilesPerRow;
            const uint64_t dStart = tileInRow * tileLength;
            const uint32_t validLength = GetValidLength(dStart);
            const uint32_t copyBytes = validLength * sizeof(DT_X);
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{
                false, 0, 0, static_cast<DT_X>(0)};

            const uint64_t firstInputOffset = sIndex * 2 * d + dStart;
            AscendC::LocalTensor<DT_X> firstLocal = inputQueue.AllocTensor<DT_X>();
            AscendC::DataCopyPad(firstLocal, xGm[firstInputOffset], copyParams, padParams);
            inputQueue.EnQue(firstLocal);
            AscendC::LocalTensor<DT_X> secondLocal = inputQueue.AllocTensor<DT_X>();
            AscendC::DataCopyPad(secondLocal, xGm[firstInputOffset + d], copyParams,
                                 padParams);
            inputQueue.EnQue(secondLocal);

            firstLocal = inputQueue.DeQue<DT_X>();
            secondLocal = inputQueue.DeQue<DT_X>();
            AscendC::LocalTensor<DT_X> oLocal = outputQueue.AllocTensor<DT_X>();
            AscendC::Add(oLocal, firstLocal, secondLocal, validLength);
            inputQueue.FreeTensor(firstLocal);
            inputQueue.FreeTensor(secondLocal);
            outputQueue.EnQue(oLocal);

            oLocal = outputQueue.DeQue<DT_X>();
            const uint64_t outputOffset = sIndex * d + dStart;
            AscendC::DataCopyPad(oGm[outputOffset], oLocal, copyParams);
            outputQueue.FreeTensor(oLocal);
        }
    }

    __aicore__ inline void ProcessBackwardHalfM2Rows(uint64_t rowStart,
                                                     uint64_t rowCount,
                                                     uint64_t alignedRowLength,
                                                     uint64_t maxBatchRows,
                                                     uint32_t inputRowGapBytes) {
        constexpr uint64_t MAX_BLOCK_COUNT = 4095;
        const uint64_t batchCapacity =
            maxBatchRows < MAX_BLOCK_COUNT ? maxBatchRows : MAX_BLOCK_COUNT;
        const uint32_t rowBytes = static_cast<uint32_t>(d * sizeof(DT_X));
        const uint8_t rightPadding = static_cast<uint8_t>(alignedRowLength - d);
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            true, 0, rightPadding, static_cast<DT_X>(0)};

        uint64_t processedRows = 0;
        while (processedRows < rowCount) {
            const uint64_t remainingRows = rowCount - processedRows;
            const uint64_t currentRows64 =
                remainingRows < batchCapacity ? remainingRows : batchCapacity;
            const uint16_t currentRows = static_cast<uint16_t>(currentRows64);
            const uint32_t activeLength =
                static_cast<uint32_t>(currentRows64 * alignedRowLength);
            const uint64_t currentRowStart = rowStart + processedRows;
            AscendC::DataCopyExtParams inputParams{
                currentRows, rowBytes, inputRowGapBytes, 0, 0};

            const uint64_t firstInputOffset = currentRowStart * 2 * d;
            AscendC::LocalTensor<DT_X> firstLocal = inputQueue.AllocTensor<DT_X>();
            AscendC::DataCopyPad(firstLocal, xGm[firstInputOffset], inputParams,
                                 padParams);
            inputQueue.EnQue(firstLocal);
            AscendC::LocalTensor<DT_X> secondLocal = inputQueue.AllocTensor<DT_X>();
            AscendC::DataCopyPad(secondLocal, xGm[firstInputOffset + d], inputParams,
                                 padParams);
            inputQueue.EnQue(secondLocal);

            firstLocal = inputQueue.DeQue<DT_X>();
            secondLocal = inputQueue.DeQue<DT_X>();
            AscendC::LocalTensor<DT_X> oLocal = outputQueue.AllocTensor<DT_X>();
            AscendC::Add(oLocal, firstLocal, secondLocal, activeLength);
            inputQueue.FreeTensor(firstLocal);
            inputQueue.FreeTensor(secondLocal);
            outputQueue.EnQue(oLocal);

            oLocal = outputQueue.DeQue<DT_X>();
            AscendC::DataCopyExtParams outputParams{currentRows, rowBytes, 0, 0, 0};
            const uint64_t outputOffset = currentRowStart * d;
            AscendC::DataCopyPad(oGm[outputOffset], oLocal, outputParams);
            outputQueue.FreeTensor(oLocal);
            processedRows += currentRows64;
        }
    }

    __aicore__ inline void ProcessBackwardRows(uint64_t rowStart, uint64_t rowCount,
                                               uint64_t alignedRowLength,
                                               uint64_t maxBatchRows,
                                               uint32_t inputRowGapBytes) {
        const uint64_t reductionExtent = GetReductionExtent();
        constexpr uint64_t MAX_BLOCK_COUNT = 4095;
        const uint64_t batchCapacity = maxBatchRows < MAX_BLOCK_COUNT ? maxBatchRows : MAX_BLOCK_COUNT;
        const uint32_t rowBytes = static_cast<uint32_t>(d * sizeof(DT_X));
        const uint8_t rightPadding = static_cast<uint8_t>(alignedRowLength - d);
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            true, 0, rightPadding, static_cast<DT_X>(0)};
        AscendC::LocalTensor<float> converted = convertedBuf.Get<float>();
        AscendC::LocalTensor<float> accumulator = accumulatorBuf.Get<float>();

        uint64_t processedRows = 0;
        while (processedRows < rowCount) {
            const uint64_t remainingRows = rowCount - processedRows;
            const uint64_t currentRows64 =
                remainingRows < batchCapacity ? remainingRows : batchCapacity;
            const uint16_t currentRows = static_cast<uint16_t>(currentRows64);
            const uint32_t activeLength =
                static_cast<uint32_t>(currentRows64 * alignedRowLength);
            const uint64_t currentRowStart = rowStart + processedRows;
            AscendC::DataCopyExtParams inputParams{
                currentRows, rowBytes, inputRowGapBytes, 0, 0};
            if (reductionExtent == 1) {
                AscendC::Duplicate(accumulator, 0.0f, activeLength);
            }
            if (reductionExtent == 1) {
                const uint64_t inputOffset = currentRowStart * d;
                AscendC::LocalTensor<DT_X> xLocal = inputQueue.AllocTensor<DT_X>();
                AscendC::DataCopyPad(xLocal, xGm[inputOffset], inputParams, padParams);
                inputQueue.EnQue(xLocal);

                xLocal = inputQueue.DeQue<DT_X>();
                AscendC::Cast(converted, xLocal, AscendC::RoundMode::CAST_NONE,
                              activeLength);
                AscendC::Add(accumulator, accumulator, converted, activeLength);
                inputQueue.FreeTensor(xLocal);
            } else {
                const uint64_t firstInputOffset = currentRowStart * reductionExtent * d;
                AscendC::LocalTensor<DT_X> firstLocal = inputQueue.AllocTensor<DT_X>();
                AscendC::DataCopyPad(firstLocal, xGm[firstInputOffset], inputParams, padParams);
                inputQueue.EnQue(firstLocal);

                for (uint64_t k = 0; k < reductionExtent; ++k) {
                    AscendC::LocalTensor<DT_X> xLocal = inputQueue.DeQue<DT_X>();
                    if (k + 1 < reductionExtent) {
                        const uint64_t nextInputOffset =
                            (currentRowStart * reductionExtent + k + 1) * d;
                        AscendC::LocalTensor<DT_X> nextLocal =
                            inputQueue.AllocTensor<DT_X>();
                        AscendC::DataCopyPad(nextLocal, xGm[nextInputOffset], inputParams,
                                             padParams);
                        inputQueue.EnQue(nextLocal);
                    }
                    if (k == 0) {
                        AscendC::Cast(accumulator, xLocal, AscendC::RoundMode::CAST_NONE,
                                      activeLength);
                    } else {
                        AscendC::Cast(converted, xLocal, AscendC::RoundMode::CAST_NONE,
                                      activeLength);
                        AscendC::Add(accumulator, accumulator, converted, activeLength);
                    }
                    inputQueue.FreeTensor(xLocal);
                }
            }

            AscendC::LocalTensor<DT_X> oLocal = outputQueue.AllocTensor<DT_X>();
            AscendC::Cast(oLocal, accumulator, AscendC::RoundMode::CAST_RINT, activeLength);
            outputQueue.EnQue(oLocal);
            oLocal = outputQueue.DeQue<DT_X>();
            AscendC::DataCopyExtParams outputParams{currentRows, rowBytes, 0, 0, 0};
            const uint64_t outputOffset = currentRowStart * d;
            AscendC::DataCopyPad(oGm[outputOffset], oLocal, outputParams);
            outputQueue.FreeTensor(oLocal);
            processedRows += currentRows64;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> forwardQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> convertedBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatorBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> oGm;
    uint64_t s;
    uint64_t d;
    uint64_t mhcMult;
    uint32_t tileLength;
    uint32_t usedCoreNum;
    uint32_t mode;
};

template <typename DT_X, uint64_t MHC_MULT_KIND>
 __global__ __aicore__ void mhc_expand(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcExpandTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcExpandTilingData, tiling_data, tiling);
    KernelMhcExpand<DT_X, MHC_MULT_KIND> op;
    op.Init(x, o, tiling_data);
    op.Process();
}
