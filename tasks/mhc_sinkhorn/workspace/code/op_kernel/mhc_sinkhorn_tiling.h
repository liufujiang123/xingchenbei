// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MhcSinkhornTilingData {
    uint64_t matrixCount;
    uint64_t matrixSize;
    uint32_t usedCoreNum;
    uint32_t iterations;
    float eps;
};
