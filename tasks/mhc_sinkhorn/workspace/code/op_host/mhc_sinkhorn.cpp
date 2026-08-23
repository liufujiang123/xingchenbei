// Host-side contract validation, specialization selection, and minimal tiling.
#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_sinkhorn_tiling.h"
#include "../op_kernel/tiling_key_mhc_sinkhorn.h"

namespace {
constexpr int64_t DEFAULT_ITERATIONS = 20;
constexpr float DEFAULT_EPS = 1.0e-6f;
constexpr int64_t MIN_ITERATIONS = 1;
constexpr int64_t MAX_ITERATIONS = 100;
constexpr uint64_t MASK_MODE_NONE = 0;
constexpr uint64_t MASK_MODE_SCALAR = 1;
constexpr uint64_t MASK_MODE_FULL = 2;

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool ResolveMatrixDomain(const gert::Shape &shape, uint64_t &matrixCount,
                         uint64_t &matrixSize, uint64_t &n) {
    const size_t rank = shape.GetDimNum();
    if (rank < 2) {
        return false;
    }
    const int64_t rowDim = shape.GetDim(rank - 2);
    const int64_t colDim = shape.GetDim(rank - 1);
    if (rowDim != colDim || (rowDim != 4 && rowDim != 6 && rowDim != 8)) {
        return false;
    }

    n = static_cast<uint64_t>(rowDim);
    if (!CheckedMul(n, n, matrixSize)) {
        return false;
    }
    matrixCount = 1;
    for (size_t axis = 0; axis + 2 < rank; ++axis) {
        const int64_t dim = shape.GetDim(axis);
        if (dim < 0 || !CheckedMul(matrixCount, static_cast<uint64_t>(dim), matrixCount)) {
            return false;
        }
    }
    return true;
}

bool ResolveAttrs(const gert::RuntimeAttrs *attrs, uint32_t &iterations, float &eps) {
    const int64_t *iterationsAttr = attrs == nullptr ? nullptr : attrs->GetInt(0);
    const float *epsAttr = attrs == nullptr ? nullptr : attrs->GetFloat(1);
    const int64_t resolvedIterations =
        iterationsAttr == nullptr ? DEFAULT_ITERATIONS : *iterationsAttr;
    const float resolvedEps = epsAttr == nullptr ? DEFAULT_EPS : *epsAttr;
    if (resolvedIterations < MIN_ITERATIONS || resolvedIterations > MAX_ITERATIONS) {
        return false;
    }
    iterations = static_cast<uint32_t>(resolvedIterations);
    eps = resolvedEps;
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor *logits = context->GetRequiredInputTensor(0);
    if (logits == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType logitsDtype = logits->GetDataType();
    if (logitsDtype != ge::DT_FLOAT16 && logitsDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    uint64_t matrixCount = 0;
    uint64_t matrixSize = 0;
    uint64_t n = 0;
    if (!ResolveMatrixDomain(logits->GetStorageShape(), matrixCount, matrixSize, n)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t expectedElementCount = 0;
    if (!CheckedMul(matrixCount, matrixSize, expectedElementCount) ||
        logits->GetShapeSize() != expectedElementCount) {
        return ge::GRAPH_FAILED;
    }

    uint64_t maskMode = MASK_MODE_NONE;
    const gert::Tensor *mask = context->GetOptionalInputTensor(1);
    if (mask != nullptr) {
        if (mask->GetDataType() != logitsDtype) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t maskElements = mask->GetShapeSize();
        if (maskElements == 1) {
            maskMode = MASK_MODE_SCALAR;
        } else if (maskElements == expectedElementCount) {
            maskMode = MASK_MODE_FULL;
        } else {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t iterations = 0;
    float eps = 0.0f;
    if (!ResolveAttrs(context->GetAttrs(), iterations, eps)) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t aivCoreCount = platform.GetCoreNumAiv();
    if (aivCoreCount <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t usedCoreNum64 = matrixCount == 0
        ? 1
        : std::min(matrixCount, static_cast<uint64_t>(aivCoreCount));
    if (usedCoreNum64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = static_cast<uint32_t>(usedCoreNum64);

    uint32_t DT_LOGITS = static_cast<uint32_t>(logitsDtype);
    const uint64_t N = n;
    const uint64_t MASK_MODE = maskMode;
    ASCENDC_TPL_SEL_PARAM(context, DT_LOGITS, N, MASK_MODE);

    MhcSinkhornTilingData *tiling = context->GetTilingData<MhcSinkhornTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->matrixCount = matrixCount;
    tiling->matrixSize = matrixSize;
    tiling->usedCoreNum = usedCoreNum;
    tiling->iterations = iterations;
    tiling->eps = eps;

    if (context->SetBlockDim(usedCoreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *logitsShape = context->GetInputShape(0);
    gert::Shape *weightsShape = context->GetOutputShape(0);
    if (logitsShape == nullptr || weightsShape == nullptr) {
        return GRAPH_FAILED;
    }
    *weightsShape = *logitsShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType logitsDtype = context->GetInputDataType(0);
    if (logitsDtype != ge::DT_FLOAT16 && logitsDtype != ge::DT_FLOAT) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, logitsDtype);
}
}  // namespace ge

namespace ops {
class MhcSinkhorn : public OpDef {
public:
    explicit MhcSinkhorn(const char *name) : OpDef(name) {
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("iterations").AttrType(OPTIONAL).Int(DEFAULT_ITERATIONS);
        this->Attr("eps").AttrType(OPTIONAL).Float(DEFAULT_EPS);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(MhcSinkhorn);
}  // namespace ops
