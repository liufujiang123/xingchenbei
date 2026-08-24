// Kernel侧核函数实现
#include <type_traits>

#include "kernel_operator.h"

#include "sparse_flash_attention_tiling.h"
#include "tiling_key_sparse_flash_attention.h"

namespace {
constexpr uint64_t CONTENT_DIM = 512;
constexpr uint64_t ROPE_DIM = 64;
constexpr uint64_t AUX_ROWS_PER_GROUP = 8;
constexpr float NEG_INF = -3.402823466e+38F;
enum class ScaleExperiment : uint32_t {
    ATTRIBUTE = 0,
    HALF_ATTRIBUTE = 1,
    INV_SQRT_512 = 2,
    INV_SQRT_576 = 3,
};
enum class CausalExperiment : uint32_t {
    RIGHT_DOWN_ACTUAL = 0,
    ORDINARY = 1,
    RIGHT_DOWN_PHYSICAL = 2,
};

constexpr bool AGGREGATE_KEY_INSTEAD_OF_VALUE = false;
constexpr ScaleExperiment SCALE_EXPERIMENT = ScaleExperiment::ATTRIBUTE;
constexpr bool ROPE_TERM_UNSCALED = false;
constexpr CausalExperiment CAUSAL_EXPERIMENT = CausalExperiment::RIGHT_DOWN_ACTUAL;
constexpr bool SPARSE_INDEX_IS_TOKEN_START = false;
constexpr bool DISABLE_ROPE_TERM = false;
constexpr bool TWO_PASS_SOFTMAX = false;
}  // namespace

template <class DT_QUERY>
class KernelSparseFlashAttention {
public:
    __aicore__ inline KernelSparseFlashAttention() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                GM_ADDR sparse_indices, GM_ADDR actual_seq_lengths_query,
                                GM_ADDR actual_seq_lengths_kv, GM_ADDR query_rope,
                                GM_ADDR key_rope, GM_ADDR attention_out,
                                GM_ADDR softmax_max_out, GM_ADDR softmax_sum_out,
                                GM_ADDR workspace,
                                const SparseFlashAttentionTilingData &tiling) {
        batchSize_ = tiling.batchSize;
        querySeqLen_ = tiling.querySeqLen;
        kvSeqLen_ = tiling.kvSeqLen;
        queryHeadNum_ = tiling.queryHeadNum;
        sparseSize_ = tiling.sparseSize;
        totalRows_ = tiling.totalRows;
        hasActualQueryLen_ = tiling.hasActualQueryLen;
        hasActualKvLen_ = tiling.hasActualKvLen;
        sparseBlockSize_ = tiling.sparseBlockSize == 0 ? 1U : tiling.sparseBlockSize;
        sparseMode_ = tiling.sparseMode;
        returnSoftmaxLse_ = tiling.returnSoftmaxLse;
        usedCoreNum_ = tiling.usedCoreNum;
        // The statement requires scaleValue to be processed at float16 precision.
        const float attributeScale = static_cast<float>(static_cast<half>(tiling.scaleValue));
        if constexpr (SCALE_EXPERIMENT == ScaleExperiment::HALF_ATTRIBUTE) {
            scaleValue_ = attributeScale * 0.5F;
        } else if constexpr (SCALE_EXPERIMENT == ScaleExperiment::INV_SQRT_512) {
            scaleValue_ = 0.04419417382415922F;
        } else if constexpr (SCALE_EXPERIMENT == ScaleExperiment::INV_SQRT_576) {
            scaleValue_ = 0.041666666666666664F;
        } else {
            scaleValue_ = attributeScale;
        }

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(query));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(key));
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(value));
        sparseIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sparse_indices));
        queryRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(query_rope));
        keyRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(key_rope));
        attentionOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_QUERY *>(attention_out));
        if (hasActualQueryLen_ != 0) {
            actualQueryLenGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_query));
        }
        if (hasActualKvLen_ != 0) {
            actualKvLenGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_kv));
        }
        if (returnSoftmaxLse_ != 0) {
            softmaxMaxOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(softmax_max_out));
            softmaxSumOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(softmax_sum_out));
        }

        pipe_.InitBuffer(queryRawBuf_, CONTENT_DIM * sizeof(DT_QUERY));
        pipe_.InitBuffer(queryFp32Buf_, CONTENT_DIM * sizeof(float));
        pipe_.InitBuffer(queryRopeRawBuf_, ROPE_DIM * sizeof(DT_QUERY));
        pipe_.InitBuffer(queryRopeFp32Buf_, ROPE_DIM * sizeof(float));
        pipe_.InitBuffer(kvRawBuf_, CONTENT_DIM * sizeof(DT_QUERY));
        pipe_.InitBuffer(kvFp32Buf_, CONTENT_DIM * sizeof(float));
        pipe_.InitBuffer(keyRopeRawBuf_, ROPE_DIM * sizeof(DT_QUERY));
        pipe_.InitBuffer(keyRopeFp32Buf_, ROPE_DIM * sizeof(float));
        pipe_.InitBuffer(accumulatorBuf_, CONTENT_DIM * sizeof(float));
        pipe_.InitBuffer(productBuf_, CONTENT_DIM * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, AUX_ROWS_PER_GROUP * sizeof(float));
        pipe_.InitBuffer(reduceWorkBuf_, CONTENT_DIM * sizeof(float));
        pipe_.InitBuffer(outputRawBuf_, CONTENT_DIM * sizeof(DT_QUERY));
        pipe_.InitBuffer(auxMaxBuf_, AUX_ROWS_PER_GROUP * sizeof(float));
        pipe_.InitBuffer(auxSumBuf_, AUX_ROWS_PER_GROUP * sizeof(float));

        queryRaw_ = queryRawBuf_.Get<DT_QUERY>();
        queryFp32_ = queryFp32Buf_.Get<float>();
        queryRopeRaw_ = queryRopeRawBuf_.Get<DT_QUERY>();
        queryRopeFp32_ = queryRopeFp32Buf_.Get<float>();
        kvRaw_ = kvRawBuf_.Get<DT_QUERY>();
        kvFp32_ = kvFp32Buf_.Get<float>();
        keyRopeRaw_ = keyRopeRawBuf_.Get<DT_QUERY>();
        keyRopeFp32_ = keyRopeFp32Buf_.Get<float>();
        accumulator_ = accumulatorBuf_.Get<float>();
        product_ = productBuf_.Get<float>();
        reduce_ = reduceBuf_.Get<float>();
        reduceWork_ = reduceWorkBuf_.Get<float>();
        outputRaw_ = outputRawBuf_.Get<DT_QUERY>();
        auxMax_ = auxMaxBuf_.Get<float>();
        auxSum_ = auxSumBuf_.Get<float>();
    }

    __aicore__ inline void Process() {
        const uint64_t coreId = static_cast<uint64_t>(AscendC::GetBlockIdx());
        const uint64_t coreCount = usedCoreNum_ == 0 ? 1U : usedCoreNum_;
        if (returnSoftmaxLse_ == 0) {
            for (uint64_t row = coreId; row < totalRows_; row += coreCount) {
                float ignoredMax = NEG_INF;
                float ignoredSum = 0.0F;
                ProcessRow(row, ignoredMax, ignoredSum);
            }
            return;
        }

        const uint64_t groupCount = (totalRows_ + AUX_ROWS_PER_GROUP - 1U) /
                                    AUX_ROWS_PER_GROUP;
        for (uint64_t group = coreId; group < groupCount; group += coreCount) {
            const uint64_t rowStart = group * AUX_ROWS_PER_GROUP;
            const uint32_t rowCount = static_cast<uint32_t>(
                totalRows_ - rowStart < AUX_ROWS_PER_GROUP
                    ? totalRows_ - rowStart
                    : AUX_ROWS_PER_GROUP);
            for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
                float rowMax = NEG_INF;
                float rowSum = 0.0F;
                ProcessRow(rowStart + localRow, rowMax, rowSum);
                auxMax_.SetValue(localRow, rowMax);
                auxSum_.SetValue(localRow, rowSum);
            }
            StoreAuxGroup(rowStart, rowCount);
        }
    }

private:
    __aicore__ inline void LoadVector(const AscendC::GlobalTensor<DT_QUERY> &source,
                                      uint64_t offset, uint32_t count,
                                      const AscendC::LocalTensor<DT_QUERY> &raw,
                                      const AscendC::LocalTensor<float> &destination) {
        // Both the raw input and the converted FP32 tile are reused by the next
        // sparse token, so complete the previous Vector consumer first.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        if constexpr (!std::is_same<DT_QUERY, float>::value) {
            AscendC::DataCopy(raw, source[offset], count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(destination, raw, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::DataCopy(destination, source[offset], count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
    }

    __aicore__ inline float Dot(const AscendC::LocalTensor<float> &left,
                                const AscendC::LocalTensor<float> &right,
                                uint32_t count) {
        AscendC::Mul(product_, left, right, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(reduce_, product_, reduceWork_, count);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        return reduce_.GetValue(0);
    }

    __aicore__ inline float ExpNegative(float value) {
        if (value >= 0.0F) {
            return 1.0F;
        }
        if (value <= -80.0F) {
            return 0.0F;
        }
        // The Vector exponential is constant-time with respect to |value| and
        // avoids the scalar baseline's O(n) repeated multiplication loop.
        AscendC::Duplicate(reduce_, value, static_cast<uint32_t>(AUX_ROWS_PER_GROUP));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(reduce_, reduce_, static_cast<uint32_t>(AUX_ROWS_PER_GROUP));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        return reduce_.GetValue(0);
    }

    __aicore__ inline uint64_t ClampLength(int32_t length, uint64_t physicalLength) const {
        if (length <= 0) {
            return 0;
        }
        const uint64_t value = static_cast<uint64_t>(length);
        return value < physicalLength ? value : physicalLength;
    }

    __aicore__ inline uint64_t GetActualQueryLen(uint64_t batch) const {
        if (hasActualQueryLen_ == 0) {
            return querySeqLen_;
        }
        return ClampLength(actualQueryLenGm_.GetValue(batch), querySeqLen_);
    }

    __aicore__ inline uint64_t GetActualKvLen(uint64_t batch) const {
        if (hasActualKvLen_ == 0) {
            return kvSeqLen_;
        }
        return ClampLength(actualKvLenGm_.GetValue(batch), kvSeqLen_);
    }

    __aicore__ inline float ComputeScore(uint64_t batch, uint64_t keyPos) {
        const uint64_t keyBase = (batch * kvSeqLen_ + keyPos) * CONTENT_DIM;
        LoadVector(keyGm_, keyBase, CONTENT_DIM, kvRaw_, kvFp32_);
        const float contentDot = Dot(queryFp32_, kvFp32_, CONTENT_DIM);

        float ropeDot = 0.0F;
        if constexpr (!DISABLE_ROPE_TERM) {
            const uint64_t keyRopeBase = (batch * kvSeqLen_ + keyPos) * ROPE_DIM;
            LoadVector(keyRopeGm_, keyRopeBase, ROPE_DIM, keyRopeRaw_, keyRopeFp32_);
            ropeDot = Dot(queryRopeFp32_, keyRopeFp32_, ROPE_DIM);
        }
        if constexpr (ROPE_TERM_UNSCALED) {
            return contentDot * scaleValue_ + ropeDot;
        }
        return (contentDot + ropeDot) * scaleValue_;
    }

    __aicore__ inline void AccumulateValue(uint64_t batch, uint64_t keyPos,
                                           float alpha, float beta, bool first) {
        const uint64_t valueBase = (batch * kvSeqLen_ + keyPos) * CONTENT_DIM;
        if constexpr (AGGREGATE_KEY_INSTEAD_OF_VALUE) {
            LoadVector(keyGm_, valueBase, CONTENT_DIM, kvRaw_, kvFp32_);
        } else {
            LoadVector(valueGm_, valueBase, CONTENT_DIM, kvRaw_, kvFp32_);
        }
        if (first) {
            AscendC::Muls(accumulator_, kvFp32_, beta, CONTENT_DIM);
        } else {
            AscendC::Muls(accumulator_, accumulator_, alpha, CONTENT_DIM);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Axpy(accumulator_, kvFp32_, beta, CONTENT_DIM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreAttention(uint64_t row, float divisor, bool zero) {
        if (zero) {
            AscendC::Duplicate(accumulator_, 0.0F, CONTENT_DIM);
        } else {
            AscendC::Muls(accumulator_, accumulator_, 1.0F / divisor, CONTENT_DIM);
        }
        AscendC::PipeBarrier<PIPE_V>();

        const uint64_t outBase = row * CONTENT_DIM;
        if constexpr (!std::is_same<DT_QUERY, float>::value) {
            constexpr AscendC::RoundMode outputRoundMode =
                std::is_same<DT_QUERY, bfloat16_t>::value
                    ? AscendC::RoundMode::CAST_RINT
                    : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(outputRaw_, accumulator_, outputRoundMode, CONTENT_DIM);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(attentionOutGm_[outBase], outputRaw_, CONTENT_DIM);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(attentionOutGm_[outBase], accumulator_, CONTENT_DIM);
        }
        // The same UB buffers are reused by the following row.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void StoreAuxGroup(uint64_t rowStart, uint32_t rowCount) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        const AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(rowCount * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(softmaxMaxOutGm_[rowStart], auxMax_, copyParams);
        AscendC::DataCopyPad(softmaxSumOutGm_[rowStart], auxSum_, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void ProcessRow(uint64_t row, float &rowMax, float &rowSum) {
        const uint64_t head = row % queryHeadNum_;
        const uint64_t queryRow = row / queryHeadNum_;
        const uint64_t queryPos = queryRow % querySeqLen_;
        const uint64_t batch = queryRow / querySeqLen_;

        const uint64_t actualQueryLen = GetActualQueryLen(batch);
        const uint64_t actualKvLen = GetActualKvLen(batch);
        if (queryPos >= actualQueryLen || actualKvLen == 0) {
            StoreAttention(row, 1.0F, true);
            rowMax = NEG_INF;
            rowSum = 0.0F;
            return;
        }

        const uint64_t queryBase =
            ((batch * querySeqLen_ + queryPos) * queryHeadNum_ + head) * CONTENT_DIM;
        const uint64_t queryRopeBase =
            ((batch * querySeqLen_ + queryPos) * queryHeadNum_ + head) * ROPE_DIM;
        LoadVector(queryGm_, queryBase, CONTENT_DIM, queryRaw_, queryFp32_);
        LoadVector(queryRopeGm_, queryRopeBase, ROPE_DIM, queryRopeRaw_, queryRopeFp32_);

        int64_t causalLimit = static_cast<int64_t>(actualKvLen - 1U);
        if (sparseMode_ == 3) {
            if constexpr (CAUSAL_EXPERIMENT == CausalExperiment::ORDINARY) {
                causalLimit = static_cast<int64_t>(queryPos);
            } else if constexpr (CAUSAL_EXPERIMENT == CausalExperiment::RIGHT_DOWN_PHYSICAL) {
                causalLimit = static_cast<int64_t>(queryPos) +
                              static_cast<int64_t>(kvSeqLen_) -
                              static_cast<int64_t>(querySeqLen_);
            } else {
                causalLimit = static_cast<int64_t>(queryPos) +
                              static_cast<int64_t>(actualKvLen) -
                              static_cast<int64_t>(actualQueryLen);
            }
        }
        const uint64_t indexBase = (batch * querySeqLen_ + queryPos) * sparseSize_;

        if constexpr (TWO_PASS_SOFTMAX) {
            bool foundScore = false;
            float fixedMax = NEG_INF;
            for (uint64_t sparseOffset = 0; sparseOffset < sparseSize_; ++sparseOffset) {
                const int64_t sparseBlockIndex = static_cast<int64_t>(
                    sparseIndicesGm_.GetValue(indexBase + sparseOffset));
                if (sparseBlockIndex < 0) {
                    break;
                }
                const uint64_t blockStart = SPARSE_INDEX_IS_TOKEN_START
                    ? static_cast<uint64_t>(sparseBlockIndex)
                    : static_cast<uint64_t>(sparseBlockIndex) *
                          static_cast<uint64_t>(sparseBlockSize_);
                if (blockStart >= actualKvLen) {
                    continue;
                }
                for (uint64_t blockOffset = 0; blockOffset < sparseBlockSize_; ++blockOffset) {
                    const uint64_t keyPos = blockStart + blockOffset;
                    if (keyPos >= actualKvLen) {
                        break;
                    }
                    if (static_cast<int64_t>(keyPos) > causalLimit) {
                        continue;
                    }
                    const float score = ComputeScore(batch, keyPos);
                    fixedMax = !foundScore || score > fixedMax ? score : fixedMax;
                    foundScore = true;
                }
            }
            if (!foundScore) {
                StoreAttention(row, 1.0F, true);
                rowMax = NEG_INF;
                rowSum = 0.0F;
                return;
            }

            bool firstValue = true;
            float fixedSum = 0.0F;
            for (uint64_t sparseOffset = 0; sparseOffset < sparseSize_; ++sparseOffset) {
                const int64_t sparseBlockIndex = static_cast<int64_t>(
                    sparseIndicesGm_.GetValue(indexBase + sparseOffset));
                if (sparseBlockIndex < 0) {
                    break;
                }
                const uint64_t blockStart = SPARSE_INDEX_IS_TOKEN_START
                    ? static_cast<uint64_t>(sparseBlockIndex)
                    : static_cast<uint64_t>(sparseBlockIndex) *
                          static_cast<uint64_t>(sparseBlockSize_);
                if (blockStart >= actualKvLen) {
                    continue;
                }
                for (uint64_t blockOffset = 0; blockOffset < sparseBlockSize_; ++blockOffset) {
                    const uint64_t keyPos = blockStart + blockOffset;
                    if (keyPos >= actualKvLen) {
                        break;
                    }
                    if (static_cast<int64_t>(keyPos) > causalLimit) {
                        continue;
                    }
                    const float weight = ExpNegative(ComputeScore(batch, keyPos) - fixedMax);
                    fixedSum += weight;
                    AccumulateValue(batch, keyPos, 1.0F, weight, firstValue);
                    firstValue = false;
                }
            }
            StoreAttention(row, fixedSum, false);
            rowMax = fixedMax;
            rowSum = fixedSum;
            return;
        }

        bool hasValue = false;
        float runningMax = NEG_INF;
        float runningSum = 0.0F;

        for (uint64_t sparseOffset = 0; sparseOffset < sparseSize_; ++sparseOffset) {
            const int64_t sparseBlockIndex = static_cast<int64_t>(
                sparseIndicesGm_.GetValue(indexBase + sparseOffset));
            // The contract's invalid entries form a suffix, so no later entry
            // can become valid after the first negative sentinel.
            if (sparseBlockIndex < 0) {
                break;
            }

            const uint64_t blockStart = SPARSE_INDEX_IS_TOKEN_START
                ? static_cast<uint64_t>(sparseBlockIndex)
                : static_cast<uint64_t>(sparseBlockIndex) *
                      static_cast<uint64_t>(sparseBlockSize_);
            if (blockStart >= actualKvLen) {
                continue;
            }

            // Each content/RoPE/value row is a contiguous GM burst. Expanding
            // a block here preserves online-softmax order without materializing
            // a block-sized K/V tile in UB.
            for (uint64_t blockOffset = 0; blockOffset < sparseBlockSize_; ++blockOffset) {
                const uint64_t keyPos = blockStart + blockOffset;
                if (keyPos >= actualKvLen) {
                    break;
                }
                if (static_cast<int64_t>(keyPos) > causalLimit) {
                    continue;
                }

                const float score = ComputeScore(batch, keyPos);
                float alpha = 0.0F;
                float beta = 1.0F;
                float newMax = score;
                if (hasValue) {
                    if (score > runningMax) {
                        alpha = ExpNegative(runningMax - score);
                    } else {
                        alpha = 1.0F;
                        beta = ExpNegative(score - runningMax);
                        newMax = runningMax;
                    }
                }
                const float newSum = runningSum * alpha + beta;
                AccumulateValue(batch, keyPos, alpha, beta, !hasValue);
                runningMax = newMax;
                runningSum = newSum;
                hasValue = true;
            }
        }

        if (!hasValue || runningSum <= 0.0F) {
            StoreAttention(row, 1.0F, true);
            rowMax = NEG_INF;
            rowSum = 0.0F;
            return;
        }

        StoreAttention(row, runningSum, false);
        rowMax = runningMax;
        rowSum = runningSum;
    }

private:
    AscendC::GlobalTensor<DT_QUERY> queryGm_;
    AscendC::GlobalTensor<DT_QUERY> keyGm_;
    AscendC::GlobalTensor<DT_QUERY> valueGm_;
    AscendC::GlobalTensor<int32_t> sparseIndicesGm_;
    AscendC::GlobalTensor<int32_t> actualQueryLenGm_;
    AscendC::GlobalTensor<int32_t> actualKvLenGm_;
    AscendC::GlobalTensor<DT_QUERY> queryRopeGm_;
    AscendC::GlobalTensor<DT_QUERY> keyRopeGm_;
    AscendC::GlobalTensor<DT_QUERY> attentionOutGm_;
    AscendC::GlobalTensor<float> softmaxMaxOutGm_;
    AscendC::GlobalTensor<float> softmaxSumOutGm_;

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> queryRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> queryFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> queryRopeRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> queryRopeFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kvRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kvFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> keyRopeRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> keyRopeFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumulatorBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> productBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> auxMaxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> auxSumBuf_;

    AscendC::LocalTensor<DT_QUERY> queryRaw_;
    AscendC::LocalTensor<float> queryFp32_;
    AscendC::LocalTensor<DT_QUERY> queryRopeRaw_;
    AscendC::LocalTensor<float> queryRopeFp32_;
    AscendC::LocalTensor<DT_QUERY> kvRaw_;
    AscendC::LocalTensor<float> kvFp32_;
    AscendC::LocalTensor<DT_QUERY> keyRopeRaw_;
    AscendC::LocalTensor<float> keyRopeFp32_;
    AscendC::LocalTensor<float> accumulator_;
    AscendC::LocalTensor<float> product_;
    AscendC::LocalTensor<float> reduce_;
    AscendC::LocalTensor<float> reduceWork_;
    AscendC::LocalTensor<DT_QUERY> outputRaw_;
    AscendC::LocalTensor<float> auxMax_;
    AscendC::LocalTensor<float> auxSum_;

    uint64_t batchSize_;
    uint64_t querySeqLen_;
    uint64_t kvSeqLen_;
    uint64_t queryHeadNum_;
    uint64_t sparseSize_;
    uint64_t totalRows_;
    uint32_t hasActualQueryLen_;
    uint32_t hasActualKvLen_;
    uint32_t sparseBlockSize_;
    uint32_t sparseMode_;
    uint32_t returnSoftmaxLse_;
    uint32_t usedCoreNum_;
    float scaleValue_;
};

template <typename DT_QUERY>
__global__ __aicore__ void sparse_flash_attention(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparse_indices,
    GM_ADDR actual_seq_lengths_query, GM_ADDR actual_seq_lengths_kv,
    GM_ADDR query_rope, GM_ADDR key_rope, GM_ADDR attention_out,
    GM_ADDR softmax_max_out, GM_ADDR softmax_sum_out,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SparseFlashAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingData, tiling_data, tiling);
    KernelSparseFlashAttention<DT_QUERY> op;
    op.Init(query, key, value, sparse_indices, actual_seq_lengths_query,
            actual_seq_lengths_kv, query_rope, key_rope, attention_out,
            softmax_max_out, softmax_sum_out, workspace, tiling_data);
    op.Process();
}
