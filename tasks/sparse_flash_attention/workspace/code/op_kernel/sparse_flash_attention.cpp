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
                                const SparseFlashAttentionTilingData &tiling) {
        query_ = reinterpret_cast<__gm__ DT_QUERY *>(query);
        key_ = reinterpret_cast<__gm__ DT_QUERY *>(key);
        value_ = reinterpret_cast<__gm__ DT_QUERY *>(value);
        sparseIndices_ = reinterpret_cast<__gm__ int32_t *>(sparse_indices);
        actualQueryLen_ = reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_query);
        actualKvLen_ = reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_kv);
        queryRopeHalf_ = reinterpret_cast<__gm__ half *>(query_rope);
        queryRopeFloat_ = reinterpret_cast<__gm__ float *>(query_rope);
        keyRopeHalf_ = reinterpret_cast<__gm__ half *>(key_rope);
        keyRopeFloat_ = reinterpret_cast<__gm__ float *>(key_rope);
        attentionOut_ = reinterpret_cast<__gm__ DT_QUERY *>(attention_out);
        softmaxMaxOut_ = reinterpret_cast<__gm__ float *>(softmax_max_out);
        softmaxSumOut_ = reinterpret_cast<__gm__ float *>(softmax_sum_out);

        batchSize_ = tiling.batchSize;
        querySeqLen_ = tiling.querySeqLen;
        kvSeqLen_ = tiling.kvSeqLen;
        queryHeadNum_ = tiling.queryHeadNum;
        sparseSize_ = tiling.sparseSize;
        totalRows_ = tiling.totalRows;
        queryRopeIsFloat_ = tiling.queryRopeIsFloat;
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
    __aicore__ inline float ReadQuery(uint64_t offset) const {
        return static_cast<float>(query_[offset]);
    }

    __aicore__ inline float ReadKey(uint64_t offset) const {
        return static_cast<float>(key_[offset]);
    }

    __aicore__ inline float ReadValue(uint64_t offset) const {
        return static_cast<float>(value_[offset]);
    }

    __aicore__ inline float ReadQueryRope(uint64_t offset) const {
        if (queryRopeIsFloat_ != 0) {
            return queryRopeFloat_[offset];
        }
        return static_cast<float>(queryRopeHalf_[offset]);
    }

    __aicore__ inline float ReadKeyRope(uint64_t offset) const {
        if (keyRopeIsFloat_ != 0) {
            return keyRopeFloat_[offset];
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

    // Accurate enough scalar exp baseline for x <= 0. Range reduction keeps the
    // polynomial on [-ln(2), 0], then applies an exact power-of-two scale by repeated 0.5.
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
        // rightDownCausal: align the right edge of the effective Q and KV sequences.
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

    __aicore__ inline void ZeroRow(uint64_t row, bool writeAux) {
        const uint64_t outBase = row * CONTENT_DIM;
        for (uint64_t d = 0; d < CONTENT_DIM; ++d) {
            attentionOut_[outBase + d] = static_cast<DT_QUERY>(0.0F);
        }
        if (writeAux) {
            softmaxMaxOut_[row] = NEG_INF;
            softmaxSumOut_[row] = 0.0F;
        }
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
            ZeroRow(row, writeAux);
            return;
        }

        const uint64_t outBase = row * CONTENT_DIM;
        for (uint64_t d = 0; d < CONTENT_DIM; ++d) {
            attentionOut_[outBase + d] = static_cast<DT_QUERY>(0.0F);
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
                const float oldAcc = hasValue ? static_cast<float>(attentionOut_[outBase + d]) : 0.0F;
                const float newAcc = oldAcc * alpha + beta * ReadValue(valueBase + d);
                attentionOut_[outBase + d] = static_cast<DT_QUERY>(newAcc);
            }

            runningMax = newMax;
            runningSum = newSum;
            hasValue = true;
        }

        if (!hasValue || runningSum <= 0.0F) {
            ZeroRow(row, writeAux);
            return;
        }

        const float invSum = 1.0F / runningSum;
        for (uint64_t d = 0; d < CONTENT_DIM; ++d) {
            const float normalized = static_cast<float>(attentionOut_[outBase + d]) * invSum;
            attentionOut_[outBase + d] = static_cast<DT_QUERY>(normalized);
        }
        if (writeAux) {
            softmaxMaxOut_[row] = runningMax;
            softmaxSumOut_[row] = runningSum;
        }
    }

private:
    __gm__ DT_QUERY *query_;
    __gm__ DT_QUERY *key_;
    __gm__ DT_QUERY *value_;
    __gm__ int32_t *sparseIndices_;
    __gm__ int32_t *actualQueryLen_;
    __gm__ int32_t *actualKvLen_;
    __gm__ half *queryRopeHalf_;
    __gm__ float *queryRopeFloat_;
    __gm__ half *keyRopeHalf_;
    __gm__ float *keyRopeFloat_;
    __gm__ DT_QUERY *attentionOut_;
    __gm__ float *softmaxMaxOut_;
    __gm__ float *softmaxSumOut_;

    uint64_t batchSize_;
    uint64_t querySeqLen_;
    uint64_t kvSeqLen_;
    uint64_t queryHeadNum_;
    uint64_t sparseSize_;
    uint64_t totalRows_;
    uint32_t queryRopeIsFloat_;
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
            softmax_max_out, softmax_sum_out, tiling_data);
    op.Process();
}
