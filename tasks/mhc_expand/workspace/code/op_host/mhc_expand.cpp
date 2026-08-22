// Host侧Tiling实现
#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_expand_tiling.h"
#include "../op_kernel/tiling_key_mhc_expand.h"

namespace {
constexpr int64_t DEFAULT_MHC_MULT = 2;
constexpr uint32_t FORWARD_MODE = 0;
constexpr uint32_t BACKWARD_MODE = 1;
constexpr uint64_t ALIGN_ELEMENTS = 16;
constexpr uint64_t FORWARD_BYTES_PER_ELEMENT = 2;
constexpr uint64_t BACKWARD_BYTES_PER_ELEMENT = 12;
constexpr uint64_t BACKWARD_PIPELINED_BYTES_PER_ELEMENT = 14;
constexpr uint64_t BACKWARD_HALF_M2_BYTES_PER_ELEMENT = 6;

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool ResolveModeAndAttrs(const gert::RuntimeAttrs *attrs, size_t rank,
                         int64_t &mhcMult, uint32_t &mode) {
    if (rank == 2) {
        mode = FORWARD_MODE;
    } else if (rank == 3) {
        mode = BACKWARD_MODE;
    } else {
        return false;
    }

    const int64_t *mhcMultAttr = attrs == nullptr ? nullptr : attrs->GetInt(0);
    mhcMult = mhcMultAttr == nullptr ? DEFAULT_MHC_MULT : *mhcMultAttr;
    if (mhcMult <= 0) {
        return false;
    }

    const bool *backwardAttr = attrs == nullptr ? nullptr : attrs->GetBool(1);
    if (backwardAttr != nullptr && static_cast<uint32_t>(*backwardAttr) != mode) {
        return false;
    }
    return true;
}

bool ValidateShapeAndProducts(const gert::Shape &shape, int64_t mhcMult,
                              uint32_t mode, uint64_t &s, uint64_t &d) {
    const int64_t sDim = shape.GetDim(0);
    const int64_t dDim = shape.GetDim(mode == FORWARD_MODE ? 1 : 2);
    if (sDim <= 0 || dDim <= 0) {
        return false;
    }
    if (mode == BACKWARD_MODE) {
        const int64_t mDim = shape.GetDim(1);
        if (mDim <= 0 || mDim != mhcMult) {
            return false;
        }
    }

    s = static_cast<uint64_t>(sDim);
    d = static_cast<uint64_t>(dDim);
    const uint64_t m = static_cast<uint64_t>(mhcMult);
    uint64_t sd = 0;
    uint64_t total = 0;
    return CheckedMul(s, d, sd) && CheckedMul(sd, m, total);
}
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }

        const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
        if (tensorX == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape &shapeX = tensorX->GetStorageShape();
        const size_t rank = shapeX.GetDimNum();

        int64_t mhcMult = DEFAULT_MHC_MULT;
        uint32_t mode = FORWARD_MODE;
        if (!ResolveModeAndAttrs(context->GetAttrs(), rank, mhcMult, mode)) {
            return ge::GRAPH_FAILED;
        }

        uint64_t s = 0;
        uint64_t d = 0;
        if (!ValidateShapeAndProducts(shapeX, mhcMult, mode, s, d)) {
            return ge::GRAPH_FAILED;
        }

        const ge::DataType dtypeX = tensorX->GetDataType();
        if (dtypeX != ge::DT_BF16 && dtypeX != ge::DT_FLOAT16) {
            return ge::GRAPH_FAILED;
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const int32_t coreNumAiv = platform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (coreNumAiv <= 0 || ubSize == 0) {
            return ge::GRAPH_FAILED;
        }

        const bool useHalfM2Path = mode == BACKWARD_MODE &&
            dtypeX == ge::DT_FLOAT16 && mhcMult == 2;
        const uint64_t bytesPerElement = mode == FORWARD_MODE
            ? FORWARD_BYTES_PER_ELEMENT
            : (useHalfM2Path
                   ? BACKWARD_HALF_M2_BYTES_PER_ELEMENT
                   : (mhcMult > 1 ? BACKWARD_PIPELINED_BYTES_PER_ELEMENT
                                  : BACKWARD_BYTES_PER_ELEMENT));
        uint64_t maxTileLength = ubSize / bytesPerElement;
        const uint64_t apiMaxTileLength =
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) / sizeof(float);
        maxTileLength = std::min(maxTileLength, apiMaxTileLength);
        const uint64_t tileLength64 = (maxTileLength / ALIGN_ELEMENTS) * ALIGN_ELEMENTS;
        if (tileLength64 == 0 || tileLength64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t tileLength = static_cast<uint32_t>(tileLength64);

        const uint64_t tilesPerRow = d / tileLength64 + (d % tileLength64 == 0 ? 0 : 1);
        uint64_t taskCount = 0;
        if (!CheckedMul(s, tilesPerRow, taskCount) || taskCount == 0) {
            return ge::GRAPH_FAILED;
        }
        constexpr uint64_t COMPLETE_ROW_TASKS_PER_CORE = 8;
        constexpr uint64_t FULL_CORE_ROW_THRESHOLD = 192;
        const uint64_t parallelTaskCount =
            tilesPerRow == 1 && taskCount < FULL_CORE_ROW_THRESHOLD
                ? (taskCount + COMPLETE_ROW_TASKS_PER_CORE - 1) /
                      COMPLETE_ROW_TASKS_PER_CORE
                : taskCount;
        const uint64_t usedCoreNum64 =
            std::min(parallelTaskCount, static_cast<uint64_t>(coreNumAiv));
        if (usedCoreNum64 == 0 || usedCoreNum64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t usedCoreNum = static_cast<uint32_t>(usedCoreNum64);

        uint32_t DT_X = static_cast<uint32_t>(dtypeX);
        const uint64_t MHC_MULT_KIND = mode == BACKWARD_MODE &&
                (mhcMult == 2 || mhcMult == 4)
            ? static_cast<uint64_t>(mhcMult)
            : 0;
        ASCENDC_TPL_SEL_PARAM(context, DT_X, MHC_MULT_KIND);

        MhcExpandTilingData *tiling = context->GetTilingData<MhcExpandTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->s = s;
        tiling->d = d;
        tiling->mhcMult = static_cast<uint64_t>(mhcMult);
        tiling->tileLength = tileLength;
        tiling->usedCoreNum = usedCoreNum;
        tiling->mode = mode;

        if (context->SetBlockDim(usedCoreNum) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
        const gert::Shape *shapeX = context->GetInputShape(0);
        gert::Shape *shapeO = context->GetOutputShape(0);
        if (shapeX == nullptr || shapeO == nullptr) {
            return GRAPH_FAILED;
        }

        const size_t rank = shapeX->GetDimNum();
        int64_t mhcMult = DEFAULT_MHC_MULT;
        uint32_t mode = FORWARD_MODE;
        if (!ResolveModeAndAttrs(context->GetAttrs(), rank, mhcMult, mode)) {
            return GRAPH_FAILED;
        }

        uint64_t s = 0;
        uint64_t d = 0;
        if (!ValidateShapeAndProducts(*shapeX, mhcMult, mode, s, d)) {
            return GRAPH_FAILED;
        }

        if (mode == FORWARD_MODE) {
            *shapeO = gert::Shape({static_cast<int64_t>(s), mhcMult, static_cast<int64_t>(d)});
        } else {
            *shapeO = gert::Shape({static_cast<int64_t>(s), static_cast<int64_t>(d)});
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
        const ge::DataType dtypeX = context->GetInputDataType(0);
        if (dtypeX != ge::DT_BF16 && dtypeX != ge::DT_FLOAT16) {
            return GRAPH_FAILED;
        }
        return context->SetOutputDataType(0, dtypeX);
    }
}  // namespace ge

namespace ops {
    class MhcExpand : public OpDef {
    public:
        explicit MhcExpand(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BF16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("o")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BF16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("mhc_mult").AttrType(OPTIONAL).Int(2);
            this->Attr("backward").AttrType(OPTIONAL).Bool();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MhcExpand);
}  // namespace ops
