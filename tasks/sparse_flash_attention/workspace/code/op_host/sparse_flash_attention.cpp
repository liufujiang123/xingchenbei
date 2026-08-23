// Host侧Tiling实现
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/sparse_flash_attention_tiling.h"
#include "../op_kernel/tiling_key_sparse_flash_attention.h"

namespace {
constexpr float DEFAULT_SCALE = 0.0884F;
constexpr int64_t DEFAULT_SPARSE_BLOCK_SIZE = 1;
constexpr int64_t DEFAULT_SPARSE_MODE = 3;
constexpr int64_t DEFAULT_ATTENTION_MODE = 2;
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

    // The generated ACLNN path has already matched this call against the
    // official OpDef. Host Tiling only extracts runtime facts needed by the
    // kernel; it must not reject an otherwise platform-legal call.
    const gert::Shape &queryShape = query->GetStorageShape();
    const gert::Shape &keyShape = key->GetStorageShape();
    const gert::Shape &sparseShape = sparseIndices->GetStorageShape();
    const uint64_t b = static_cast<uint64_t>(queryShape.GetDim(0));
    const uint64_t qs = static_cast<uint64_t>(queryShape.GetDim(1));
    const uint64_t qn = static_cast<uint64_t>(queryShape.GetDim(2));
    const uint64_t kvs = static_cast<uint64_t>(keyShape.GetDim(1));
    const uint64_t sparseSize = static_cast<uint64_t>(sparseShape.GetDim(3));

    const ge::DataType queryDtype = query->GetDataType();
    const ge::DataType queryRopeDtype = queryRope->GetDataType();
    const ge::DataType keyRopeDtype = keyRope->GetDataType();

    float scaleValue = DEFAULT_SCALE;
    int64_t sparseBlockSize = DEFAULT_SPARSE_BLOCK_SIZE;
    int64_t sparseMode = DEFAULT_SPARSE_MODE;
    int64_t attentionMode = DEFAULT_ATTENTION_MODE;
    bool returnSoftmaxLse = false;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float *scale = attrs->GetFloat(0);
        const int64_t *blockSize = attrs->GetInt(1);
        const int64_t *mode = attrs->GetInt(2);
        const int64_t *attnMode = attrs->GetInt(3);
        const bool *returnLse = attrs->GetBool(4);
        scaleValue = scale == nullptr ? DEFAULT_SCALE : *scale;
        sparseBlockSize = blockSize == nullptr ? DEFAULT_SPARSE_BLOCK_SIZE : *blockSize;
        sparseMode = mode == nullptr ? DEFAULT_SPARSE_MODE : *mode;
        attentionMode = attnMode == nullptr ? DEFAULT_ATTENTION_MODE : *attnMode;
        returnSoftmaxLse = returnLse == nullptr ? false : *returnLse;
    }
    const uint64_t totalRows = b * qs * qn;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t reportedCoreNumAiv = platform.GetCoreNumAiv();
    // The generic CANN 8.5 Ascend910B platform record reports zero standalone
    // VectorCore units even though the contest binary is an AIV kernel. Do not
    // reject otherwise legal inputs because that platform metadata is coarse;
    // one block is a correctness-safe fallback on the actual 910B target.
    const uint32_t usableCoreNumAiv = reportedCoreNumAiv == 0 ? 1U : reportedCoreNumAiv;
    // Scalar auxiliary writes share cache lines across adjacent rows. Keep the
    // correctness baseline single-core when those optional outputs are enabled.
    const uint64_t usedCore64 = returnSoftmaxLse
        ? 1U
        : std::min<uint64_t>(totalRows, static_cast<uint64_t>(usableCoreNumAiv));
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
    tiling->primaryIsBf16 = queryDtype == ge::DT_BF16 ? 1U : 0U;
    tiling->primaryIsFloat = queryDtype == ge::DT_FLOAT ? 1U : 0U;
    tiling->queryRopeIsBf16 = queryRopeDtype == ge::DT_BF16 ? 1U : 0U;
    tiling->queryRopeIsFloat = queryRopeDtype == ge::DT_FLOAT ? 1U : 0U;
    tiling->keyRopeIsBf16 = keyRopeDtype == ge::DT_BF16 ? 1U : 0U;
    tiling->keyRopeIsFloat = keyRopeDtype == ge::DT_FLOAT ? 1U : 0U;
    tiling->hasActualQueryLen = actualQueryLen == nullptr ? 0U : 1U;
    tiling->hasActualKvLen = actualKvLen == nullptr ? 0U : 1U;
    tiling->sparseBlockSize = static_cast<uint32_t>(sparseBlockSize);
    tiling->sparseMode = static_cast<uint32_t>(sparseMode);
    tiling->attentionMode = static_cast<uint32_t>(attentionMode);
    tiling->returnSoftmaxLse = returnSoftmaxLse ? 1U : 0U;
    tiling->usedCoreNum = usedCoreNum;
    tiling->scaleValue = scaleValue;

    context->SetBlockDim(usedCoreNum);
    size_t *workspace = context->GetWorkspaceSizes(1);
    // The kernel keeps its accumulator in per-core UB; workspace is unused.
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    return GRAPH_SUCCESS;
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
