// Kernel侧核函数实现
#include "kernel_operator.h"

#include "sparse_flash_attention_tiling.h"
#include "tiling_key_sparse_flash_attention.h"

namespace {
constexpr uint64_t CONTENT_DIM = 512;
constexpr uint64_t ROPE_DIM = 64;
constexpr float NEG_INF = -3.402823466e+38F;
constexpr float LN2 = 0.6931471805599453F;
constexpr float INV_LN2 = 1.4426950408889634F;
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
        queryHalf_ = reinterpret_cast<__gm__ half *>(query);
        queryRaw_ = reinterpret_cast<__gm__ uint16_t *>(query);
        queryFloat_ = reinterpret_cast<__gm__ float *>(query);
        keyHalf_ = reinterpret_cast<__gm__ half *>(key);
        keyRaw_ = reinterpret_cast<__gm__ uint16_t *>(key);
        keyFloat_ = reinterpret_cast<__gm__ float *>(key);
        valueHalf_ = reinterpret_cast<__gm__ half *>(value);
        valueRaw_ = reinterpret_cast<__gm__ uint16_t *>(value);
        valueFloat_ = reinterpret_cast<__gm__ float *>(value);
        sparseIndices_ = reinterpret_cast<__gm__ int32_t *>(sparse_indices);
        actualQueryLen_ = reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_query);
        actualKvLen_ = reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_kv);
        queryRopeHalf_ = reinterpret_cast<__gm__ half *>(query_rope);
        queryRopeRaw_ = reinterpret_cast<__gm__ uint16_t *>(query_rope);
        queryRopeFloat_ = reinterpret_cast<__gm__ float *>(query_rope);
        keyRopeHalf_ = reinterpret_cast<__gm__ half *>(key_rope);
        keyRopeRaw_ = reinterpret_cast<__gm__ uint16_t *>(key_rope);
        keyRopeFloat_ = reinterpret_cast<__gm__ float *>(key_rope);
        attentionOutHalf_ = reinterpret_cast<__gm__ half *>(attention_out);
        attentionOutRaw_ = reinterpret_cast<__gm__ uint16_t *>(attention_out);
        attentionOutFloat_ = reinterpret_cast<__gm__ float *>(attention_out);
        softmaxMaxOut_ = reinterpret_cast<__gm__ float *>(softmax_max_out);
        softmaxSumOut_ = reinterpret_cast<__gm__ float *>(softmax_sum_out);
        pipe_.InitBuffer(accBuf_, CONTENT_DIM * sizeof(float));
        accumulator_ = accBuf_.Get<float>();

        batchSize_ = tiling.batchSize;
        querySeqLen_ = tiling.querySeqLen;
        kvSeqLen_ = tiling.kvSeqLen;
        queryHeadNum_ = tiling.queryHeadNum;
        sparseSize_ = tiling.sparseSize;
        totalRows_ = tiling.totalRows;
        primaryIsBf16_ = tiling.primaryIsBf16;
        primaryIsFloat_ = tiling.primaryIsFloat;
        queryRopeIsBf16_ = tiling.queryRopeIsBf16;
        queryRopeIsFloat_ = tiling.queryRopeIsFloat;
        keyRopeIsBf16_ = tiling.keyRopeIsBf16;
        keyRopeIsFloat_ = tiling.keyRopeIsFloat;
        hasActualQueryLen_ = tiling.hasActualQueryLen;
        hasActualKvLen_ = tiling.hasActualKvLen;
        sparseMode_ = tiling.sparseMode;
        returnSoftmaxLse_ = tiling.returnSoftmaxLse;
        usedCoreNum_ = tiling.usedCoreNum;

        // The statement requires scaleValue to be processed at float16 precision.
        scaleValue_ = static_cast<float>(static_cast<half>(tiling.scaleValue));
    }

    __aicore__ inline void Process() {
        const uint64_t coreId = static_cast<uint64_t>(AscendC::GetBlockIdx());
        const uint64_t coreCount = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        for (uint64_t row = coreId; row < totalRows_; row += coreCount) {
            ProcessRow(row);
        }
    }

private:
    template <typename T>
    __aicore__ inline void FlushCacheLine(__gm__ T *address) const {
        AscendC::GlobalTensor<T> line;
        line.SetGlobalBuffer(address, 1);
        AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(line);
    }

    __aicore__ inline void FlushOutputs(uint64_t row, bool writeAux) const {
        const uint64_t outBase = row * CONTENT_DIM;
        constexpr uint64_t FLUSH_STRIDE_BYTES = 32;
        const uint64_t flushStrideElements = primaryIsFloat_ != 0
            ? FLUSH_STRIDE_BYTES / sizeof(float)
            : FLUSH_STRIDE_BYTES / sizeof(uint16_t);
        for (uint64_t d = 0; d < CONTENT_DIM; d += flushStrideElements) {
            if (primaryIsFloat_ != 0) {
                FlushCacheLine(attentionOutFloat_ + outBase + d);
            } else {
                FlushCacheLine(attentionOutRaw_ + outBase + d);
            }
        }
        if (writeAux) {
            FlushCacheLine(softmaxMaxOut_ + row);
            FlushCacheLine(softmaxSumOut_ + row);
        }
    }

    __aicore__ inline float Bf16BitsToFloat(uint16_t value) const {
        union {
            uint32_t bits;
            float scalar;
        } converted;
        converted.bits = static_cast<uint32_t>(value) << 16;
        return converted.scalar;
    }

    __aicore__ inline uint16_t FloatToBf16Bits(float value) const {
        union {
            float scalar;
            uint32_t bits;
        } converted;
        converted.scalar = value;
        const uint32_t exponent = converted.bits & 0x7F800000U;
        const uint32_t mantissa = converted.bits & 0x007FFFFFU;
        if (exponent == 0x7F800000U && mantissa != 0U) {
            return static_cast<uint16_t>((converted.bits >> 16) | 0x0040U);
        }
        const uint32_t roundingBias = 0x00007FFFU + ((converted.bits >> 16) & 1U);
        return static_cast<uint16_t>((converted.bits + roundingBias) >> 16);
    }

    __aicore__ inline float ReadQuery(uint64_t offset) const {
        if (primaryIsFloat_ != 0) {
            return queryFloat_[offset];
        }
        return primaryIsBf16_ != 0 ? Bf16BitsToFloat(queryRaw_[offset])
                                   : static_cast<float>(queryHalf_[offset]);
    }

    __aicore__ inline float ReadKey(uint64_t offset) const {
        if (primaryIsFloat_ != 0) {
            return keyFloat_[offset];
        }
        return primaryIsBf16_ != 0 ? Bf16BitsToFloat(keyRaw_[offset])
                                   : static_cast<float>(keyHalf_[offset]);
    }

    __aicore__ inline float ReadValue(uint64_t offset) const {
        if (primaryIsFloat_ != 0) {
            return valueFloat_[offset];
        }
        return primaryIsBf16_ != 0 ? Bf16BitsToFloat(valueRaw_[offset])
                                   : static_cast<float>(valueHalf_[offset]);
    }

    __aicore__ inline float ReadQueryRope(uint64_t offset) const {
        if (queryRopeIsFloat_ != 0) {
            return queryRopeFloat_[offset];
        }
        if (queryRopeIsBf16_ != 0) {
            return Bf16BitsToFloat(queryRopeRaw_[offset]);
        }
        return static_cast<float>(queryRopeHalf_[offset]);
    }

    __aicore__ inline float ReadKeyRope(uint64_t offset) const {
        if (keyRopeIsFloat_ != 0) {
            return keyRopeFloat_[offset];
        }
        if (keyRopeIsBf16_ != 0) {
            return Bf16BitsToFloat(keyRopeRaw_[offset]);
        }
        return static_cast<float>(keyRopeHalf_[offset]);
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
        return ClampLength(actualQueryLen_[batch], querySeqLen_);
    }

    __aicore__ inline uint64_t GetActualKvLen(uint64_t batch) const {
        if (hasActualKvLen_ == 0) {
            return kvSeqLen_;
        }
        return ClampLength(actualKvLen_[batch], kvSeqLen_);
    }

    // Scalar correctness baseline for exp(x), x <= 0. Range reduction keeps the
    // polynomial on [-ln(2), 0], then applies a power-of-two scale.
    __aicore__ inline float ExpNegative(float x) const {
        if (x >= 0.0F) {
            return 1.0F;
        }
        if (x <= -80.0F) {
            return 0.0F;
        }
        const int32_t n = static_cast<int32_t>((-x) * INV_LN2);
        const float r = x + static_cast<float>(n) * LN2;
        const float poly = 1.0F + r * (1.0F + r * (0.5F + r * (0.1666666666666667F +
            r * (0.0416666666666667F + r * (0.0083333333333333F +
            r * (0.0013888888888889F + r * (0.0001984126984127F +
            r * 0.0000248015873016F)))))));
        float scale = 1.0F;
        for (int32_t i = 0; i < n; ++i) {
            scale *= 0.5F;
        }
        return poly * scale;
    }

    __aicore__ inline bool PassCausal(uint64_t queryPos, int64_t keyPos,
                                      uint64_t queryLen, uint64_t kvLen) const {
        if (sparseMode_ != 3) {
            return true;
        }
        const int64_t limit = static_cast<int64_t>(queryPos) +
                              static_cast<int64_t>(kvLen) -
                              static_cast<int64_t>(queryLen);
        return keyPos <= limit;
    }

    __aicore__ inline float ComputeScore(uint64_t batch, uint64_t queryPos,
                                         uint64_t head, uint64_t keyPos) const {
        const uint64_t queryBase = ((batch * querySeqLen_ + queryPos) * queryHeadNum_ + head) * CONTENT_DIM;
        const uint64_t keyBase = (batch * kvSeqLen_ + keyPos) * CONTENT_DIM;
        float dot = 0.0F;
        for (uint64_t d = 0; d < CONTENT_DIM; ++d) {
            dot += ReadQuery(queryBase + d) * ReadKey(keyBase + d);
        }

        const uint64_t queryRopeBase = ((batch * querySeqLen_ + queryPos) * queryHeadNum_ + head) * ROPE_DIM;
        const uint64_t keyRopeBase = (batch * kvSeqLen_ + keyPos) * ROPE_DIM;
        for (uint64_t d = 0; d < ROPE_DIM; ++d) {
            dot += ReadQueryRope(queryRopeBase + d) * ReadKeyRope(keyRopeBase + d);
        }
        return dot * scaleValue_;
    }

    __aicore__ inline void StoreAttention(uint64_t offset, float value) {
        if (primaryIsFloat_ != 0) {
            attentionOutFloat_[offset] = value;
        } else if (primaryIsBf16_ != 0) {
            attentionOutRaw_[offset] = FloatToBf16Bits(value);
        } else {
            attentionOutHalf_[offset] = static_cast<half>(value);
        }
    }

    __aicore__ inline void StoreZeroOutput(uint64_t row, bool writeAux) {
        const uint64_t outBase = row * CONTENT_DIM;
        for (uint64_t d = 0; d < CONTENT_DIM; ++d) {
            StoreAttention(outBase + d, 0.0F);
        }
        if (writeAux) {
            softmaxMaxOut_[row] = NEG_INF;
            softmaxSumOut_[row] = 0.0F;
        }
        FlushOutputs(row, writeAux);
    }

    __aicore__ inline void ProcessRow(uint64_t row) {
        const uint64_t head = row % queryHeadNum_;
        const uint64_t queryRow = row / queryHeadNum_;
        const uint64_t queryPos = queryRow % querySeqLen_;
        const uint64_t batch = queryRow / querySeqLen_;
        const bool writeAux = returnSoftmaxLse_ != 0;

        const uint64_t actualQueryLen = GetActualQueryLen(batch);
        const uint64_t actualKvLen = GetActualKvLen(batch);
        if (queryPos >= actualQueryLen || actualKvLen == 0) {
            StoreZeroOutput(row, writeAux);
            return;
        }

        for (uint64_t d = 0; d < CONTENT_DIM; ++d) {
            accumulator_.SetValue(d, 0.0F);
        }

        const uint64_t indexBase = (batch * querySeqLen_ + queryPos) * sparseSize_;
        bool hasValue = false;
        float runningMax = NEG_INF;
        float runningSum = 0.0F;

        for (uint64_t k = 0; k < sparseSize_; ++k) {
            const int64_t keyPosSigned = static_cast<int64_t>(sparseIndices_[indexBase + k]);
            if (keyPosSigned < 0 || keyPosSigned >= static_cast<int64_t>(actualKvLen) ||
                !PassCausal(queryPos, keyPosSigned, actualQueryLen, actualKvLen)) {
                continue;
            }
            const uint64_t keyPos = static_cast<uint64_t>(keyPosSigned);
            const float score = ComputeScore(batch, queryPos, head, keyPos);

            float alpha = 0.0F;
            float beta = 1.0F;
            float newMax = score;
            if (hasValue) {
                if (score > runningMax) {
                    alpha = ExpNegative(runningMax - score);
                    beta = 1.0F;
                    newMax = score;
                } else {
                    alpha = 1.0F;
                    beta = ExpNegative(score - runningMax);
                    newMax = runningMax;
                }
            }
            const float newSum = runningSum * alpha + beta;
            const uint64_t valueBase = (batch * kvSeqLen_ + keyPos) * CONTENT_DIM;
            for (uint64_t d = 0; d < CONTENT_DIM; ++d) {
                const float oldAcc = accumulator_.GetValue(d);
                accumulator_.SetValue(d, oldAcc * alpha + beta * ReadValue(valueBase + d));
            }

            runningMax = newMax;
            runningSum = newSum;
            hasValue = true;
        }

        if (!hasValue || runningSum <= 0.0F) {
            StoreZeroOutput(row, writeAux);
            return;
        }

        const float invSum = 1.0F / runningSum;
        const uint64_t outBase = row * CONTENT_DIM;
        for (uint64_t d = 0; d < CONTENT_DIM; ++d) {
            StoreAttention(outBase + d, accumulator_.GetValue(d) * invSum);
        }
        if (writeAux) {
            softmaxMaxOut_[row] = runningMax;
            softmaxSumOut_[row] = runningSum;
        }
        FlushOutputs(row, writeAux);
    }

private:
    __gm__ half *queryHalf_;
    __gm__ uint16_t *queryRaw_;
    __gm__ float *queryFloat_;
    __gm__ half *keyHalf_;
    __gm__ uint16_t *keyRaw_;
    __gm__ float *keyFloat_;
    __gm__ half *valueHalf_;
    __gm__ uint16_t *valueRaw_;
    __gm__ float *valueFloat_;
    __gm__ int32_t *sparseIndices_;
    __gm__ int32_t *actualQueryLen_;
    __gm__ int32_t *actualKvLen_;
    __gm__ half *queryRopeHalf_;
    __gm__ uint16_t *queryRopeRaw_;
    __gm__ float *queryRopeFloat_;
    __gm__ half *keyRopeHalf_;
    __gm__ uint16_t *keyRopeRaw_;
    __gm__ float *keyRopeFloat_;
    __gm__ half *attentionOutHalf_;
    __gm__ uint16_t *attentionOutRaw_;
    __gm__ float *attentionOutFloat_;
    __gm__ float *softmaxMaxOut_;
    __gm__ float *softmaxSumOut_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::LocalTensor<float> accumulator_;

    uint64_t batchSize_;
    uint64_t querySeqLen_;
    uint64_t kvSeqLen_;
    uint64_t queryHeadNum_;
    uint64_t sparseSize_;
    uint64_t totalRows_;
    uint32_t primaryIsBf16_;
    uint32_t primaryIsFloat_;
    uint32_t queryRopeIsBf16_;
    uint32_t queryRopeIsFloat_;
    uint32_t keyRopeIsBf16_;
    uint32_t keyRopeIsFloat_;
    uint32_t hasActualQueryLen_;
    uint32_t hasActualKvLen_;
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
