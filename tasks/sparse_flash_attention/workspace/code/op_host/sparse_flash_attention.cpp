// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/sparse_flash_attention_tiling.h"
#include "../op_kernel/tiling_key_sparse_flash_attention.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_query = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_key = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_value = context->GetRequiredInputTensor(2);
        const gert::Tensor *tensor_sparse_indices = context->GetRequiredInputTensor(3);
        const gert::Tensor *tensor_actual_seq_lengths_query = context->GetOptionalInputTensor(4);
        if(tensor_actual_seq_lengths_query) {
            // 可选输入数组存在
        }
        const gert::Tensor *tensor_actual_seq_lengths_kv = context->GetOptionalInputTensor(5);
        if(tensor_actual_seq_lengths_kv) {
            // 可选输入数组存在
        }
        const gert::Tensor *tensor_query_rope = context->GetRequiredInputTensor(6);
        const gert::Tensor *tensor_key_rope = context->GetRequiredInputTensor(7);
        ge::DataType dtype_query = tensor_query->GetDataType(); // 获取数据类型
        int dtype_size_query = ge::GetSizeByDataType(dtype_query); // 获取数据类型的字长
        uint32_t length_query = tensor_query->GetShapeSize(); // 获取元素个数
        uint32_t size_query = tensor_query->GetSize(); // 获取内存大小
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_scale_value = attrs->GetFloat(0);
        const int64_t *attr_sparse_block_size = attrs->GetInt(1);
        const int64_t *attr_sparse_mode = attrs->GetInt(2);
        const int64_t *attr_attention_mode = attrs->GetInt(3);
        const bool *attr_return_softmax_lse = attrs->GetBool(4);
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_QUERY = static_cast<uint32_t>(dtype_query);
        ASCENDC_TPL_SEL_PARAM(context, DT_QUERY);
        // 示例: 计算tiling方案并填充tiling结构体
        SparseFlashAttentionTilingData *tiling = context->GetTilingData<SparseFlashAttentionTilingData>();
        tiling->length = length_query;
        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
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
