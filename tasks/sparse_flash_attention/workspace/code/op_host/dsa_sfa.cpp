// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/dsa_sfa_tiling.h"
#include "../op_kernel/tiling_key_dsa_sfa.h"

namespace {
uint64_t ShapeSize(const gert::StorageShape *shape) {
    return shape == nullptr ? 0 : static_cast<uint64_t>(shape->GetStorageShape().GetShapeSize());
}

uint32_t LastDim(const gert::StorageShape *shape) {
    if (shape == nullptr) {
        return 0;
    }
    const auto &s = shape->GetStorageShape();
    const size_t rank = s.GetDimNum();
    if (rank == 0) {
        return 0;
    }
    const int64_t dim = s.GetDim(rank - 1);
    return dim > 0 ? static_cast<uint32_t>(dim) : 0;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *valuesShape = context->GetInputShape(0);
    const gert::StorageShape *indexShape = context->GetInputShape(1);
    const gert::StorageShape *gateShape = context->GetInputShape(2);
    const gert::StorageShape *scoreShape = context->GetInputShape(3);
    const gert::StorageShape *aggregatedShape = context->GetOutputShape(0);
    const gert::StorageShape *aggWeightsShape = context->GetOutputShape(1);
    if (valuesShape == nullptr || indexShape == nullptr || gateShape == nullptr || scoreShape == nullptr ||
        aggregatedShape == nullptr || aggWeightsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto &valuesStorage = valuesShape->GetStorageShape();
    const auto &indexStorage = indexShape->GetStorageShape();
    const auto &scoreStorage = scoreShape->GetStorageShape();
    if (valuesStorage.GetDimNum() == 0 || indexStorage.GetDimNum() == 0 || scoreStorage.GetDimNum() == 0) {
        return ge::GRAPH_FAILED;
    }

    DsaSfaTilingData *tiling = context->GetTilingData<DsaSfaTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->valuesNumel = ShapeSize(valuesShape);
    tiling->sparseIndexNumel = ShapeSize(indexShape);
    tiling->gateNumel = ShapeSize(gateShape);
    tiling->scoreNumel = ShapeSize(scoreShape);
    tiling->aggregatedNumel = ShapeSize(aggregatedShape);
    tiling->aggWeightsNumel = ShapeSize(aggWeightsShape);

    const int64_t batchDim = valuesStorage.GetDim(0);
    tiling->batchSize = batchDim > 0 ? static_cast<uint32_t>(batchDim) : 0;
    tiling->valueDim = LastDim(valuesShape);
    tiling->sparseSize = LastDim(scoreShape);
    tiling->sourceRowsPerBatch = 0;
    tiling->indexRowsPerBatch = 0;
    tiling->scoreRowsPerBatch = 0;
    tiling->headBroadcast = 0;
    tiling->mode = 0;

    const auto *indexDesc = context->GetInputDesc(1);
    const auto *gateDesc = context->GetInputDesc(2);
    const auto *scoreDesc = context->GetInputDesc(3);
    const auto *valuesDesc = context->GetInputDesc(0);
    if (indexDesc == nullptr || gateDesc == nullptr || scoreDesc == nullptr || valuesDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType valuesDtype = valuesDesc->GetDataType();
    const ge::DataType indexDtype = indexDesc->GetDataType();
    const ge::DataType gateDtype = gateDesc->GetDataType();
    const ge::DataType scoreDtype = scoreDesc->GetDataType();

    tiling->sparseIndexIsInt64 = indexDtype == ge::DT_INT64 ? 1U : 0U;
    tiling->gateIsFloat = gateDtype == ge::DT_FLOAT ? 1U : 0U;
    tiling->scoreIsFloat = scoreDtype == ge::DT_FLOAT ? 1U : 0U;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *scale = attrs == nullptr ? nullptr : attrs->GetFloat(0);
    tiling->scale = scale == nullptr ? 1.0F : *scale;

    const uint32_t indexSparseSize = LastDim(indexShape);
    const bool basicShapeOk = tiling->batchSize > 0 && tiling->valueDim > 0 && tiling->sparseSize > 0 &&
                              indexSparseSize == tiling->sparseSize && tiling->gateNumel == tiling->scoreNumel;

    uint64_t sourceRows = 0;
    uint64_t indexRows = 0;
    uint64_t scoreRows = 0;
    if (basicShapeOk) {
        const uint64_t valueDenom = static_cast<uint64_t>(tiling->batchSize) * tiling->valueDim;
        if (valueDenom > 0 && tiling->valuesNumel % valueDenom == 0 &&
            tiling->sparseIndexNumel % tiling->sparseSize == 0 && tiling->scoreNumel % tiling->sparseSize == 0) {
            sourceRows = tiling->valuesNumel / valueDenom;
            indexRows = tiling->sparseIndexNumel / tiling->sparseSize;
            scoreRows = tiling->scoreNumel / tiling->sparseSize;
        }
    }

    const bool rowLayoutOk = sourceRows > 0 && indexRows > 0 && scoreRows > 0 &&
                             indexRows % tiling->batchSize == 0 && scoreRows % tiling->batchSize == 0 &&
                             scoreRows % indexRows == 0;

    if (rowLayoutOk) {
        tiling->sourceRowsPerBatch = static_cast<uint32_t>(sourceRows);
        tiling->indexRowsPerBatch = static_cast<uint32_t>(indexRows / tiling->batchSize);
        tiling->scoreRowsPerBatch = static_cast<uint32_t>(scoreRows / tiling->batchSize);
        tiling->headBroadcast = static_cast<uint32_t>(scoreRows / indexRows);

        const uint64_t expectedAggregated = scoreRows * tiling->valueDim;
        const uint64_t expectedWeights = tiling->scoreNumel;
        if (tiling->aggregatedNumel == expectedAggregated && tiling->aggWeightsNumel == expectedWeights) {
            tiling->mode = 1;
        }
    }

    // The probe is scalar and correctness-oriented. Multiple cores are used only when each
    // output row is at least one cache line, avoiding scalar SetValue/GetValue cache-line races.
    uint32_t blockDim = 1;
    if (tiling->mode == 1) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t numCoresAiv = platform.GetCoreNumAiv();
        const int valueBytes = ge::GetSizeByDataType(valuesDtype);
        const int scoreBytes = ge::GetSizeByDataType(scoreDtype);
        const bool rowCacheSafe = valueBytes > 0 && scoreBytes > 0 &&
                                  static_cast<uint64_t>(tiling->valueDim) * valueBytes >= 64 &&
                                  static_cast<uint64_t>(tiling->sparseSize) * scoreBytes >= 64;
        if (rowCacheSafe && numCoresAiv > 1) {
            uint64_t totalRows = static_cast<uint64_t>(tiling->batchSize) * tiling->scoreRowsPerBatch;
            blockDim = totalRows < static_cast<uint64_t>(numCoresAiv) ? static_cast<uint32_t>(totalRows)
                                                                      : static_cast<uint32_t>(numCoresAiv);
            if (blockDim == 0) {
                blockDim = 1;
            }
        }
    }
    tiling->blockDim = blockDim;
    context->SetBlockDim(blockDim);

    uint32_t DT_VALUES = static_cast<uint32_t>(valuesDtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_VALUES);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *valuesShape = context->GetInputShape(0);
    const gert::Shape *scoreShape = context->GetInputShape(3);
    gert::Shape *aggregatedShape = context->GetOutputShape(0);
    gert::Shape *aggWeightsShape = context->GetOutputShape(1);
    if (valuesShape == nullptr || scoreShape == nullptr || aggregatedShape == nullptr || aggWeightsShape == nullptr ||
        valuesShape->GetDimNum() == 0 || scoreShape->GetDimNum() == 0) {
        return GRAPH_FAILED;
    }

    // Probe-v1 hypothesis: score's last axis is sparse K; aggregation replaces K with value D.
    *aggregatedShape = *scoreShape;
    aggregatedShape->SetDim(scoreShape->GetDimNum() - 1, valuesShape->GetDim(valuesShape->GetDimNum() - 1));
    *aggWeightsShape = *scoreShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(3));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class DsaSfa : public OpDef {
public:
    explicit DsaSfa(const char *name) : OpDef(name) {
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("sparse_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("score")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("aggregated")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("agg_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(DsaSfa);
}  // namespace ops
