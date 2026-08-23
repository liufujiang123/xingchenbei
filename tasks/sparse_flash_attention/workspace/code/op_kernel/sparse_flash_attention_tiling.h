// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct SparseFlashAttentionTilingData {
    uint64_t batchSize;
    uint64_t querySeqLen;
    uint64_t kvSeqLen;
    uint64_t queryHeadNum;
    uint64_t sparseSize;
    uint64_t totalRows;

    uint32_t primaryIsBf16;
    uint32_t primaryIsFloat;
    uint32_t queryRopeIsBf16;
    uint32_t queryRopeIsFloat;
    uint32_t keyRopeIsBf16;
    uint32_t keyRopeIsFloat;
    uint32_t hasActualQueryLen;
    uint32_t hasActualKvLen;
    uint32_t sparseBlockSize;
    uint32_t sparseMode;
    uint32_t attentionMode;
    uint32_t returnSoftmaxLse;
    uint32_t usedCoreNum;

    float scaleValue;
};
