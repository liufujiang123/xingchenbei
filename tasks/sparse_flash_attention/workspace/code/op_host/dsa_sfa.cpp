// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/dsa_sfa_tiling.h"
#include "../op_kernel/tiling_key_dsa_sfa.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_values = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_sparse_index = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_gate = context->GetRequiredInputTensor(2);
        const gert::Tensor *tensor_score = context->GetRequiredInputTensor(3);
        ge::DataType dtype_values = tensor_values->GetDataType(); // 获取数据类型
        int dtype_size_values = ge::GetSizeByDataType(dtype_values); // 获取数据类型的字长
        uint32_t length_values = tensor_values->GetShapeSize(); // 获取元素个数
        uint32_t size_values = tensor_values->GetSize(); // 获取内存大小
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_scale = attrs->GetFloat(0);
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_VALUES = static_cast<uint32_t>(dtype_values);
        ASCENDC_TPL_SEL_PARAM(context, DT_VALUES);
        // 示例: 计算tiling方案并填充tiling结构体
        DsaSfaTilingData *tiling = context->GetTilingData<DsaSfaTilingData>();
        tiling->length = length_values;
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
