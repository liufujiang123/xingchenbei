// 910B direct-ACLNN diagnostic Host control. This is intentionally not part of
// workspace/code and is used only to decide whether 561002 happens before our
// production TilingFunc can matter.
#include <cstdint>

#include "register/op_def_registry.h"

#include "../op_kernel/sparse_flash_attention_tiling.h"
#include "../op_kernel/tiling_key_sparse_flash_attention.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor *query = context->GetRequiredInputTensor(0);
    const gert::Tensor *queryRope = context->GetRequiredInputTensor(6);
    const gert::Tensor *keyRope = context->GetRequiredInputTensor(7);
    if (query == nullptr || queryRope == nullptr || keyRope == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t dtype = static_cast<uint32_t>(query->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dtype);
    auto *tiling = context->GetTilingData<SparseFlashAttentionTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->batchSize = 1;
    tiling->querySeqLen = 1;
    tiling->kvSeqLen = 1;
    tiling->queryHeadNum = 1;
    tiling->sparseSize = 1;
    tiling->totalRows = 1;
    tiling->primaryIsBf16 = query->GetDataType() == ge::DT_BF16 ? 1U : 0U;
    tiling->primaryIsFloat = query->GetDataType() == ge::DT_FLOAT ? 1U : 0U;
    tiling->queryRopeIsBf16 = queryRope->GetDataType() == ge::DT_BF16 ? 1U : 0U;
    tiling->queryRopeIsFloat = queryRope->GetDataType() == ge::DT_FLOAT ? 1U : 0U;
    tiling->keyRopeIsBf16 = keyRope->GetDataType() == ge::DT_BF16 ? 1U : 0U;
    tiling->keyRopeIsFloat = keyRope->GetDataType() == ge::DT_FLOAT ? 1U : 0U;
    tiling->hasActualQueryLen = context->GetOptionalInputTensor(4) == nullptr ? 0U : 1U;
    tiling->hasActualKvLen = context->GetOptionalInputTensor(5) == nullptr ? 0U : 1U;
    tiling->sparseBlockSize = 1;
    tiling->sparseMode = 3;
    tiling->attentionMode = 2;
    tiling->returnSoftmaxLse = 0;
    tiling->usedCoreNum = 1;
    tiling->scaleValue = 0.0884F;
    context->SetBlockDim(1);
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *) { return GRAPH_SUCCESS; }
static graphStatus InferDataType(gert::InferDataTypeContext *) { return GRAPH_SUCCESS; }
}  // namespace ge

namespace ops {
class SparseFlashAttention : public OpDef {
public:
    explicit SparseFlashAttention(const char *name) : OpDef(name) {
        this->Input("query").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("sparse_indices").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_seq_lengths_query").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_seq_lengths_kv").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("query_rope").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key_rope").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attention_out").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_max_out").ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_sum_out").ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0884);
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("attention_mode").AttrType(OPTIONAL).Int(2);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SparseFlashAttention);
}  // namespace ops
