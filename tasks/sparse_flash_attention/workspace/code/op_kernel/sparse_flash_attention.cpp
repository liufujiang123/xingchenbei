// Kernel侧核函数实现
#include "kernel_operator.h"

#include "sparse_flash_attention_tiling.h"
#include "tiling_key_sparse_flash_attention.h"

template <class DT_QUERY>
class KernelSparseFlashAttention {
public:
    __aicore__ inline KernelSparseFlashAttention() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparse_indices, GM_ADDR actual_seq_lengths_query, GM_ADDR actual_seq_lengths_kv, GM_ADDR query_rope, GM_ADDR key_rope, GM_ADDR attention_out, GM_ADDR softmax_max_out, GM_ADDR softmax_sum_out, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_QUERY>
 __global__ __aicore__ void sparse_flash_attention(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparse_indices, GM_ADDR actual_seq_lengths_query, GM_ADDR actual_seq_lengths_kv, GM_ADDR query_rope, GM_ADDR key_rope, GM_ADDR attention_out, GM_ADDR softmax_max_out, GM_ADDR softmax_sum_out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SparseFlashAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingData, tiling_data, tiling);
    KernelSparseFlashAttention<DT_QUERY> op;
    op.Init(query, key, value, sparse_indices, actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope, attention_out, softmax_max_out, softmax_sum_out, tiling_data.length);
    op.Process();
}
