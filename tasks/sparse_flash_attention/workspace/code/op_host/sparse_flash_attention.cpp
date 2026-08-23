// Host侧Tiling实现
#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/sparse_flash_attention_tiling.h"
#include "../op_kernel/tiling_key_sparse_flash_attention.h"

namespace {
constexpr int64_t CONTENT_DIM = 512;
constexpr int64_t ROPE_DIM = 64;
constexpr int64_t KV_HEAD_NUM = 1;
constexpr float DEFAULT_SCALE = 0.0884F;
constexpr int64_t DEFAULT_SPARSE_BLOCK_SIZE = 1;
constexpr int64_t DEFAULT_SPARSE_MODE = 3;
constexpr int64_t DEFAULT_ATTENTION_MODE = 2;

bool IsFloatTensorType(ge::DataType dtype) {
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT;
}

bool IsPowerOfTwo(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

bool ReadRank4Shape(const gert::Tensor *tensor, int64_t &d0, int64_t &d1,
                    int64_t &d2, int64_t &d3) {
    if (tensor == nullptr) {
        return false;
    }
    const gert::Shape &shape = tensor->GetStorageShape();
    if (shape.GetDimNum() != 4) {
        return false;
    }
    d0 = shape.GetDim(0);
    d1 = shape.GetDim(1);
    d2 = shape.GetDim(2);
    d3 = shape.GetDim(3);
    return d0 > 0 && d1 > 0 && d2 > 0 && d3 > 0;
}

bool ReadAttrs(const gert::RuntimeAttrs *attrs, float &scaleValue,
               int64_t &sparseBlockSize, int64_t &sparseMode,
               int64_t &attentionMode, bool &returnSoftmaxLse) {
    scaleValue = DEFAULT_SCALE;
    sparseBlockSize = DEFAULT_SPARSE_BLOCK_SIZE;
    sparseMode = DEFAULT_SPARSE_MODE;
    attentionMode = DEFAULT_ATTENTION_MODE;
    returnSoftmaxLse = false;

    if (attrs == nullptr) {
        return true;
    }
    const float *scale = attrs->GetFloat(0);
    const int64_t *blockSize = attrs->GetInt(1);
    const int64_t *mode = attrs->GetInt(2);
    const int64_t *attnMode = attrs->GetInt(3);
    const bool *returnLse = attrs->GetBool(4);
    if (scale != nullptr) {
        scaleValue = *scale;
    }
    if (blockSize != nullptr) {
        sparseBlockSize = *blockSize;
    }
    if (mode != nullptr) {
        sparseMode = *mode;
    }
    if (attnMode != nullptr) {
        attentionMode = *attnMode;
    }
    if (returnLse != nullptr) {
        returnSoftmaxLse = *returnLse;
    }

    if (!IsPowerOfTwo(sparseBlockSize) || sparseBlockSize > 128) {
        return false;
    }
    if (sparseMode != 0 && sparseMode != 3) {
        return false;
    }
    return attentionMode == 2;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *query = context->GetRequiredInputTensor(0);
    const gert::Tensor *key = context->GetRequiredInputTensor(1);
    const gert::Tensor *value = context->GetRequiredInputTensor(2);
    const gert::Tensor *sparseIndices = context->GetRequiredInputTensor(3);
    const gert::Tensor *actualQueryLen = context->GetOptionalInputTensor(4);
    const gert::Tensor *actualKvLen = context->GetOptionalInputTensor(5);
    const gert::Tensor *queryRope = context->GetRequiredInputTensor(6);
    const gert::Tensor *keyRope = context->GetRequiredInputTensor(7);
    if (query == nullptr || key == nullptr || value == nullptr || sparseIndices == nullptr ||
        queryRope == nullptr || keyRope == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t b = 0;
    int64_t qs = 0;
    int64_t qn = 0;
    int64_t qd = 0;
    int64_t kb = 0;
    int64_t kvs = 0;
    int64_t kvn = 0;
    int64_t kd = 0;
    int64_t vb = 0;
    int64_t vks = 0;
    int64_t vkn = 0;
    int64_t vd = 0;
    int64_t ib = 0;
    int64_t iqs = 0;
    int64_t ikvn = 0;
    int64_t sparseSize = 0;
    int64_t qrb = 0;
    int64_t qrqs = 0;
    int64_t qrqn = 0;
    int64_t qrd = 0;
    int64_t krb = 0;
    int64_t krks = 0;
    int64_t krn = 0;
    int64_t krd = 0;

    if (!ReadRank4Shape(query, b, qs, qn, qd) ||
        !ReadRank4Shape(key, kb, kvs, kvn, kd) ||
        !ReadRank4Shape(value, vb, vks, vkn, vd) ||
        !ReadRank4Shape(sparseIndices, ib, iqs, ikvn, sparseSize) ||
        !ReadRank4Shape(queryRope, qrb, qrqs, qrqn, qrd) ||
        !ReadRank4Shape(keyRope, krb, krks, krn, krd)) {
        return ge::GRAPH_FAILED;
    }

    if (b != kb || b != vb || b != ib || b != qrb || b != krb ||
        qs != iqs || qs != qrqs || qn != qrqn ||
        kvs != vks || kvs != krks ||
        qd != CONTENT_DIM || kd != CONTENT_DIM || vd != CONTENT_DIM ||
        qrd != ROPE_DIM || krd != ROPE_DIM ||
        kvn != KV_HEAD_NUM || vkn != KV_HEAD_NUM || ikvn != KV_HEAD_NUM || krn != KV_HEAD_NUM ||
        sparseSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType queryDtype = query->GetDataType();
    const ge::DataType keyDtype = key->GetDataType();
    const ge::DataType valueDtype = value->GetDataType();
    const ge::DataType sparseDtype = sparseIndices->GetDataType();
    const ge::DataType queryRopeDtype = queryRope->GetDataType();
    const ge::DataType keyRopeDtype = keyRope->GetDataType();
    if (!IsFloatTensorType(queryDtype) || !IsFloatTensorType(keyDtype) ||
        !IsFloatTensorType(valueDtype) || !IsFloatTensorType(queryRopeDtype) ||
        !IsFloatTensorType(keyRopeDtype) || sparseDtype != ge::DT_INT32 ||
        queryDtype != keyDtype || queryDtype != valueDtype) {
        return ge::GRAPH_FAILED;
    }

    if (actualQueryLen != nullptr) {
        const gert::Shape &shape = actualQueryLen->GetStorageShape();
        if (shape.GetDimNum() != 1 || shape.GetDim(0) != b ||
            actualQueryLen->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }
    if (actualKvLen != nullptr) {
        const gert::Shape &shape = actualKvLen->GetStorageShape();
        if (shape.GetDimNum() != 1 || shape.GetDim(0) != b ||
            actualKvLen->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    float scaleValue = DEFAULT_SCALE;
    int64_t sparseBlockSize = DEFAULT_SPARSE_BLOCK_SIZE;
    int64_t sparseMode = DEFAULT_SPARSE_MODE;
    int64_t attentionMode = DEFAULT_ATTENTION_MODE;
    bool returnSoftmaxLse = false;
    if (!ReadAttrs(context->GetAttrs(), scaleValue, sparseBlockSize, sparseMode,
                   attentionMode, returnSoftmaxLse)) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t ub = static_cast<uint64_t>(b);
    const uint64_t uqs = static_cast<uint64_t>(qs);
    const uint64_t uqn = static_cast<uint64_t>(qn);
    if (ub > std::numeric_limits<uint64_t>::max() / uqs) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t bq = ub * uqs;
    if (bq > std::numeric_limits<uint64_t>::max() / uqn) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalRows = bq * uqn;
    if (totalRows == 0) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t coreNumAiv = platform.GetCoreNumAiv();
    if (coreNumAiv <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t usedCore64 = std::min<uint64_t>(totalRows, static_cast<uint64_t>(coreNumAiv));
    const uint32_t usedCoreNum = static_cast<uint32_t>(usedCore64);

    uint32_t DT_QUERY = static_cast<uint32_t>(queryDtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_QUERY);

    SparseFlashAttentionTilingData *tiling = context->GetTilingData<SparseFlashAttentionTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->batchSize = static_cast<uint64_t>(b);
    tiling->querySeqLen = static_cast<uint64_t>(qs);
    tiling->kvSeqLen = static_cast<uint64_t>(kvs);
    tiling->queryHeadNum = static_cast<uint64_t>(qn);
    tiling->sparseSize = static_cast<uint64_t>(sparseSize);
    tiling->totalRows = totalRows;
    tiling->queryRopeIsFloat = queryRopeDtype == ge::DT_FLOAT ? 1U : 0U;
    tiling->keyRopeIsFloat = keyRopeDtype == ge::DT_FLOAT ? 1U : 0U;
    tiling->hasActualQueryLen = actualQueryLen == nullptr ? 0U : 1U;
    tiling->hasActualKvLen = actualKvLen == nullptr ? 0U : 1U;
    tiling->sparseBlockSize = static_cast<uint32_t>(sparseBlockSize);
    tiling->sparseMode = static_cast<uint32_t>(sparseMode);
    tiling->attentionMode = static_cast<uint32_t>(attentionMode);
    tiling->returnSoftmaxLse = returnSoftmaxLse ? 1U : 0U;
    tiling->usedCoreNum = usedCoreNum;
    tiling->scaleValue = scaleValue;

    if (context->SetBlockDim(usedCoreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t workspaceBytes = usedCore64 * static_cast<uint64_t>(CONTENT_DIM) * sizeof(float);
    if (workspaceBytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = static_cast<size_t>(workspaceBytes);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(0);
    gert::Shape *attentionShape = context->GetOutputShape(0);
    if (queryShape == nullptr || attentionShape == nullptr || queryShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    *attentionShape = *queryShape;

    const int64_t b = queryShape->GetDim(0);
    const int64_t qs = queryShape->GetDim(1);
    const int64_t qn = queryShape->GetDim(2);
    if (b <= 0 || qs <= 0 || qn <= 0) {
        return GRAPH_FAILED;
    }
    const gert::Shape auxShape({b, 1, qs, qn});
    gert::Shape *maxShape = context->GetOutputShape(1);
    gert::Shape *sumShape = context->GetOutputShape(2);
    if (maxShape != nullptr) {
        *maxShape = auxShape;
    }
    if (sumShape != nullptr) {
        *sumShape = auxShape;
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType queryDtype = context->GetInputDataType(0);
    if (!IsFloatTensorType(queryDtype)) {
        return GRAPH_FAILED;
    }
    if (context->SetOutputDataType(0, queryDtype) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (context->SetOutputDataType(1, ge::DT_FLOAT) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(2, ge::DT_FLOAT);
}
}  // namespace ge

namespace ops {
class SparseFlashAttention : public OpDef {
public:
    explicit SparseFlashAttention(const char *name) : OpDef(name) {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("query_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_max_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_sum_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0884);
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("attention_mode").AttrType(OPTIONAL).Int(2);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(SparseFlashAttention);
}  // namespace ops
