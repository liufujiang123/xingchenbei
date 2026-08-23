// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// Probe-v1 contract hypothesis:
//   values       : [B, KV, ..., D] value table, flattened to B * sourceRowsPerBatch * D
//   sparse_index : [B, ..., K] sparse KV indices, shared across query heads when needed
//   gate         : same flattened shape as score; non-positive entries are masked
//   score        : [B, ..., H, K] precomputed attention logits
//   aggregated   : score leading dimensions with K replaced by D
//   agg_weights  : same shape as score
//
// This is intentionally a falsifiable evaluator probe, not a claim about undocumented
// platform semantics. The public OpDef itself remains unchanged.
struct DsaSfaTilingData {
    uint64_t valuesNumel;
    uint64_t sparseIndexNumel;
    uint64_t gateNumel;
    uint64_t scoreNumel;
    uint64_t aggregatedNumel;
    uint64_t aggWeightsNumel;

    uint32_t batchSize;
    uint32_t valueDim;
    uint32_t sparseSize;
    uint32_t sourceRowsPerBatch;
    uint32_t indexRowsPerBatch;
    uint32_t scoreRowsPerBatch;
    uint32_t headBroadcast;
    uint32_t blockDim;

    uint32_t mode;
    uint32_t sparseIndexIsInt64;
    uint32_t gateIsFloat;
    uint32_t scoreIsFloat;
    float scale;
};
