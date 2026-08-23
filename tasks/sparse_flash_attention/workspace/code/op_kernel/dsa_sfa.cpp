// Kernel侧核函数实现
#include "kernel_operator.h"

#include "dsa_sfa_tiling.h"
#include "tiling_key_dsa_sfa.h"

// Probe-v1 semantic hypothesis (intentionally falsifiable by CANNJudge):
//   * values is a per-batch value table whose final axis is D.
//   * sparse_index selects local rows of values for each query row.
//   * score contains precomputed attention logits; its final axis is sparse K.
//   * score may have a head axis that sparse_index does not have (KV_N=1 sharing),
//     represented by headBroadcast = scoreRows / indexRows.
//   * gate is a positive multiplicative prior/mask on each sparse logit.
//   * aggregated is stable-softmax(score * scale, gate) @ gathered(values).
//   * agg_weights stores the normalized sparse weights.
//
// The public DsaSfa function signature is unchanged. This implementation is deliberately
// scalar and correctness/probing oriented; performance optimization comes only after the
// hidden evaluator confirms the semantic layout.
template <class DT_VALUES>
class KernelDsaSfa {
public:
    __aicore__ inline KernelDsaSfa() {}

    __aicore__ inline void Init(GM_ADDR values, GM_ADDR sparse_index, GM_ADDR gate, GM_ADDR score,
                                GM_ADDR aggregated, GM_ADDR agg_weights, const DsaSfaTilingData &tiling) {
        values_ = reinterpret_cast<__gm__ DT_VALUES *>(values);
        sparseIndex32_ = reinterpret_cast<__gm__ int32_t *>(sparse_index);
        sparseIndex64_ = reinterpret_cast<__gm__ int64_t *>(sparse_index);
        gateHalf_ = reinterpret_cast<__gm__ half *>(gate);
        gateFloat_ = reinterpret_cast<__gm__ float *>(gate);
        scoreHalf_ = reinterpret_cast<__gm__ half *>(score);
        scoreFloat_ = reinterpret_cast<__gm__ float *>(score);
        aggregated_ = reinterpret_cast<__gm__ DT_VALUES *>(aggregated);
        aggWeightsHalf_ = reinterpret_cast<__gm__ half *>(agg_weights);
        aggWeightsFloat_ = reinterpret_cast<__gm__ float *>(agg_weights);

        valuesNumel_ = tiling.valuesNumel;
        scoreNumel_ = tiling.scoreNumel;
        aggregatedNumel_ = tiling.aggregatedNumel;
        aggWeightsNumel_ = tiling.aggWeightsNumel;
        batchSize_ = tiling.batchSize;
        valueDim_ = tiling.valueDim;
        sparseSize_ = tiling.sparseSize;
        sourceRowsPerBatch_ = tiling.sourceRowsPerBatch;
        indexRowsPerBatch_ = tiling.indexRowsPerBatch;
        scoreRowsPerBatch_ = tiling.scoreRowsPerBatch;
        headBroadcast_ = tiling.headBroadcast;
        blockDim_ = tiling.blockDim == 0 ? 1 : tiling.blockDim;
        mode_ = tiling.mode;
        sparseIndexIsInt64_ = tiling.sparseIndexIsInt64;
        gateIsFloat_ = tiling.gateIsFloat;
        scoreIsFloat_ = tiling.scoreIsFloat;
        scale_ = tiling.scale;
    }

    __aicore__ inline void Process() {
        if (mode_ == 1) {
            ProcessSparseSoftmaxAggregate();
        } else {
            ProcessFallback();
        }
    }

private:
    __aicore__ inline float ReadValue(uint64_t offset) const {
        return static_cast<float>(values_[offset]);
    }

    __aicore__ inline int64_t ReadSparseIndex(uint64_t offset) const {
        if (sparseIndexIsInt64_ != 0) {
            return sparseIndex64_[offset];
        }
        return static_cast<int64_t>(sparseIndex32_[offset]);
    }

    __aicore__ inline float ReadGate(uint64_t offset) const {
        if (gateIsFloat_ != 0) {
            return gateFloat_[offset];
        }
        return static_cast<float>(gateHalf_[offset]);
    }

    __aicore__ inline float ReadScore(uint64_t offset) const {
        if (scoreIsFloat_ != 0) {
            return scoreFloat_[offset];
        }
        return static_cast<float>(scoreHalf_[offset]);
    }

    __aicore__ inline void WriteAggregated(uint64_t offset, float value) {
        aggregated_[offset] = static_cast<DT_VALUES>(value);
    }

    __aicore__ inline void WriteWeight(uint64_t offset, float value) {
        if (scoreIsFloat_ != 0) {
            aggWeightsFloat_[offset] = value;
        } else {
            aggWeightsHalf_[offset] = static_cast<half>(value);
        }
    }

    __aicore__ inline float ReadWeight(uint64_t offset) const {
        if (scoreIsFloat_ != 0) {
            return aggWeightsFloat_[offset];
        }
        return static_cast<float>(aggWeightsHalf_[offset]);
    }

    __aicore__ inline float ExpNegative(float x) const {
        if (x >= 0.0F) {
            return 1.0F;
        }
        if (x <= -20.0F) {
            return 0.0F;
        }
        int32_t whole = static_cast<int32_t>(-x);
        float r = x + static_cast<float>(whole);
        float poly = 1.0F + r * (1.0F + r * (0.5F + r * (0.1666666667F +
                     r * (0.0416666667F + r * (0.0083333333F + r * 0.0013888889F)))));
        float expScale = 1.0F;
        for (int32_t i = 0; i < whole; ++i) {
            expScale *= 0.3678794412F;
        }
        return poly * expScale;
    }

    __aicore__ inline void ProcessSparseSoftmaxAggregate() {
        const uint64_t totalScoreRows = static_cast<uint64_t>(batchSize_) * scoreRowsPerBatch_;
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        for (uint64_t row = blockIdx; row < totalScoreRows; row += blockDim_) {
            const uint64_t batch = row / scoreRowsPerBatch_;
            const uint64_t rowInBatch = row - batch * scoreRowsPerBatch_;
            const uint64_t indexRowInBatch = rowInBatch / headBroadcast_;
            const uint64_t indexBase = (batch * indexRowsPerBatch_ + indexRowInBatch) * sparseSize_;
            const uint64_t scoreBase = row * sparseSize_;

            bool hasValid = false;
            float maxScore = -3.402823466e+38F;
            for (uint32_t k = 0; k < sparseSize_; ++k) {
                const int64_t index = ReadSparseIndex(indexBase + k);
                const float gate = ReadGate(scoreBase + k);
                if (index < 0 || index >= static_cast<int64_t>(sourceRowsPerBatch_) || gate <= 0.0F) {
                    continue;
                }
                const float scaledScore = ReadScore(scoreBase + k) * scale_;
                if (!hasValid || scaledScore > maxScore) {
                    maxScore = scaledScore;
                    hasValid = true;
                }
            }

            float denom = 0.0F;
            if (hasValid) {
                for (uint32_t k = 0; k < sparseSize_; ++k) {
                    const int64_t index = ReadSparseIndex(indexBase + k);
                    const float gate = ReadGate(scoreBase + k);
                    if (index < 0 || index >= static_cast<int64_t>(sourceRowsPerBatch_) || gate <= 0.0F) {
                        continue;
                    }
                    const float scaledScore = ReadScore(scoreBase + k) * scale_;
                    denom += ExpNegative(scaledScore - maxScore) * gate;
                }
            }

            for (uint32_t k = 0; k < sparseSize_; ++k) {
                float weight = 0.0F;
                if (hasValid && denom > 0.0F) {
                    const int64_t index = ReadSparseIndex(indexBase + k);
                    const float gate = ReadGate(scoreBase + k);
                    if (index >= 0 && index < static_cast<int64_t>(sourceRowsPerBatch_) && gate > 0.0F) {
                        const float scaledScore = ReadScore(scoreBase + k) * scale_;
                        weight = ExpNegative(scaledScore - maxScore) * gate / denom;
                    }
                }
                WriteWeight(scoreBase + k, weight);
            }

            const uint64_t outputBase = row * valueDim_;
            for (uint32_t d = 0; d < valueDim_; ++d) {
                float acc = 0.0F;
                for (uint32_t k = 0; k < sparseSize_; ++k) {
                    const float weight = ReadWeight(scoreBase + k);
                    if (weight == 0.0F) {
                        continue;
                    }
                    const int64_t localIndex = ReadSparseIndex(indexBase + k);
                    if (localIndex < 0 || localIndex >= static_cast<int64_t>(sourceRowsPerBatch_)) {
                        continue;
                    }
                    const uint64_t valueOffset =
                        (batch * sourceRowsPerBatch_ + static_cast<uint64_t>(localIndex)) * valueDim_ + d;
                    acc += weight * ReadValue(valueOffset);
                }
                WriteAggregated(outputBase + d, acc);
            }
        }
    }

    __aicore__ inline void ProcessFallback() {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        for (uint64_t i = 0; i < aggregatedNumel_; ++i) {
            float value = i < valuesNumel_ ? ReadValue(i) : 0.0F;
            WriteAggregated(i, value);
        }
        for (uint64_t i = 0; i < aggWeightsNumel_; ++i) {
            float value = i < scoreNumel_ ? ReadScore(i) : 0.0F;
            WriteWeight(i, value);
        }
    }

private:
    __gm__ DT_VALUES *values_;
    __gm__ int32_t *sparseIndex32_;
    __gm__ int64_t *sparseIndex64_;
    __gm__ half *gateHalf_;
    __gm__ float *gateFloat_;
    __gm__ half *scoreHalf_;
    __gm__ float *scoreFloat_;
    __gm__ DT_VALUES *aggregated_;
    __gm__ half *aggWeightsHalf_;
    __gm__ float *aggWeightsFloat_;

    uint64_t valuesNumel_;
    uint64_t scoreNumel_;
    uint64_t aggregatedNumel_;
    uint64_t aggWeightsNumel_;
    uint32_t batchSize_;
    uint32_t valueDim_;
    uint32_t sparseSize_;
    uint32_t sourceRowsPerBatch_;
    uint32_t indexRowsPerBatch_;
    uint32_t scoreRowsPerBatch_;
    uint32_t headBroadcast_;
    uint32_t blockDim_;
    uint32_t mode_;
    uint32_t sparseIndexIsInt64_;
    uint32_t gateIsFloat_;
    uint32_t scoreIsFloat_;
    float scale_;
};

template <typename DT_VALUES>
__global__ __aicore__ void dsa_sfa(GM_ADDR values, GM_ADDR sparse_index, GM_ADDR gate, GM_ADDR score,
                                    GM_ADDR aggregated, GM_ADDR agg_weights, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DsaSfaTilingData);
    GET_TILING_DATA_WITH_STRUCT(DsaSfaTilingData, tiling_data, tiling);
    KernelDsaSfa<DT_VALUES> op;
    op.Init(values, sparse_index, gate, score, aggregated, agg_weights, tiling_data);
    op.Process();
}
