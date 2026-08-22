// Kernel侧核函数实现
#include "kernel_operator.h"

#include "dsa_sfa_tiling.h"
#include "tiling_key_dsa_sfa.h"

template <class DT_VALUES>
class KernelDsaSfa {
public:
    __aicore__ inline KernelDsaSfa() {}
    __aicore__ inline void Init(GM_ADDR values, GM_ADDR sparse_index, GM_ADDR gate, GM_ADDR score, GM_ADDR aggregated, GM_ADDR agg_weights, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_VALUES>
 __global__ __aicore__ void dsa_sfa(GM_ADDR values, GM_ADDR sparse_index, GM_ADDR gate, GM_ADDR score, GM_ADDR aggregated, GM_ADDR agg_weights, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DsaSfaTilingData);
    GET_TILING_DATA_WITH_STRUCT(DsaSfaTilingData, tiling_data, tiling);
    KernelDsaSfa<DT_VALUES> op;
    op.Init(values, sparse_index, gate, score, aggregated, agg_weights, tiling_data.length);
    op.Process();
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            