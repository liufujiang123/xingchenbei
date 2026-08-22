// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MhcExpandTilingData {
    uint64_t s;
    uint64_t d;
    uint64_t mhcMult;
    uint32_t tileLength;
    uint32_t usedCoreNum;
    uint32_t mode;
};
