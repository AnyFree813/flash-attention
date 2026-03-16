import sys
import os
import torch
import torch_npu
import pytest
import flash_attn_2_cuda
from flash_attn import flash_attn_with_kvcache

def group_matmul(head, kv_head, left, right, high_prec = 1):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        if high_prec == 0:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32)).to(torch.float32)
        else:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32))
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return score

def softmax1( 
    qk_result,
    is_first,
    gm,
    interm_dtype = torch.float16
    ):
    sim = qk_result.to(interm_dtype)
    lm = torch.max(sim, dim=-1, keepdims=True)[0]
    if is_first:
        hm = lm
        dm = 0
    else:
        hm = torch.maximum(gm, lm)
        dm = gm - hm
    gm = hm
    sim_sub = sim - hm
    sim_sub = torch.exp(sim_sub.to(interm_dtype))
    row_sum = torch.sum(sim_sub, dim=-1, keepdims=True)
    return sim_sub, row_sum, dm, gm

def qkMM1( 
    query,
    key
    ):
    result = None
    qk_k = key.shape[1]
    qk_k_split = 128
    qk_k_loop = (qk_k + 127) // 128
    for qk_k_loop_idx in range(qk_k_loop):
        sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
        partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
        partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def pvMM2( 
    p,
    value
    ):
    result = None
    pv_k = value.shape[1]
    pv_k_split = 128
    pv_k_loop = (pv_k + 127) // 128
    for pv_k_loop_idx in range(pv_k_loop):
        sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)
        partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
        partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :] 
        result_split = group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def ref_flash_attention( 
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    alibi_slopes=None,
    key_leftpad=None
    ):
    inner_prec = 0
    interm_dtype = torch.float16 if inner_prec == 1 else torch.float32
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    scale = torch.tensor(scale)
    scale = scale.to(torch.float16) if inner_prec == 1 else scale.to(torch.float32)
    context_len = key.shape[2]
    context_size = 512
    group_num = query.shape[0] // key.shape[0]
    gl = None
    gl_high = None
    go = None
    go_high = None
    if mask is not None:
        mask = mask.cpu()
    for kv_start in range(0, context_len, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        sub_mask = None
        if mask is not None:
            sub_mask = mask[:query.shape[1], kv_start : kv_start + sub_len].to(interm_dtype) * (-1e4)
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        qk_result = qkMM1(query, sub_key).to(interm_dtype)
        qk_result = qk_result * scale
        if mask is not None:
            qk_result += sub_mask
        # Apply alibi mask if provided
        if alibi_slopes is not None:
            batch_size, q_seqlen, num_heads, head_size = query.shape[1], query.shape[1], query.shape[0], query.shape[2]
            alibi = torch.zeros(q_seqlen, sub_len, device=qk_result.device, dtype=interm_dtype)
            for i in range(q_seqlen):
                for j in range(sub_len):
                    # 计算实际的键位置，考虑左对齐偏移
                    actual_key_pos = j + kv_start
                    if key_leftpad is not None:
                        actual_key_pos -= key_leftpad
                    alibi[i, j] = -abs(i - actual_key_pos)
            alibi = alibi.unsqueeze(0).repeat(num_heads, 1, 1)
            alibi_slopes_iter = alibi_slopes.unsqueeze(1).unsqueeze(2)
            alibi = alibi * alibi_slopes_iter
            qk_result += alibi

        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        p_result = p_result.to(data_type)
        if kv_start == 0:
            gm_high = None
        lo = pvMM2(p_result, sub_value).to(interm_dtype)
        if kv_start == 0:
            gl = row_sum
            go = lo
        else:
            dm = torch.exp(dm)
            gl = gl * dm
            gl = gl + row_sum
            go = go * dm
            go = go + lo
    go = go / gl
    go = go.permute(1, 0, 2)
    lse = torch.squeeze((torch.log(gl) + gm), dim=-1).to(torch.float32)
    return go.to(data_type), lse

test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, mask_type)
    # 基础测试用例
    (torch.bfloat16, 1, 1, 1, 512, 512, 64, 0, 128, 0),
    (torch.bfloat16, 1, 1, 1, 512, 512, 64, 0, 128, 1),
    (torch.bfloat16, 1, 1, 1, 512, 512, 64, 0, 128, 2),
    (torch.bfloat16, 1, 4, 4, 128, 128, 128, 0, 128, 2),
    (torch.bfloat16, 1, 4, 4, 256, 256, 128, 0, 128, 2),
    (torch.bfloat16, 1, 4, 4, 512, 512, 128, 0, 128, 2),
    (torch.bfloat16, 1, 4, 4, 1024, 1024, 128, 0, 128, 2),
    (torch.bfloat16, 1, 4, 4, 2048, 2048, 128, 0, 128, 2),
    (torch.bfloat16, 1, 4, 4, 2049, 2049, 128, 0, 128, 2),
    
    # 不同数据类型
    (torch.float16, 1, 1, 1, 512, 512, 64, 0, 128, 2),
    
    # 不同 batch size
    (torch.float16, 1, 4, 4, 256, 256, 128, 0, 128, 2),
    (torch.float16, 2, 4, 4, 256, 256, 128, 0, 128, 2),
    (torch.float16, 4, 4, 4, 256, 256, 128, 0, 128, 2),
    
    # 不同的 num_heads 和 kv_heads 组合
    (torch.float16, 1, 8, 1, 256, 256, 64, 0, 128, 2),  # 8个注意力头共享1个KV头
    (torch.float16, 1, 8, 2, 256, 256, 64, 0, 128, 2),  # 8个注意力头共享2个KV头
    (torch.float16, 1, 8, 4, 256, 256, 64, 0, 128, 2),  # 8个注意力头共享4个KV头
    (torch.float16, 1, 8, 8, 256, 256, 64, 0, 128, 2),  # 每个注意力头有自己的KV头
    
    # 不同的序列长度
    (torch.float16, 1, 4, 4, 128, 128, 128, 0, 128, 2),
    (torch.float16, 1, 4, 4, 256, 256, 128, 0, 128, 2),
    (torch.float16, 1, 4, 4, 512, 512, 128, 0, 128, 2),
    (torch.float16, 1, 4, 4, 1024, 1024, 128, 0, 128, 2),
    (torch.float16, 1, 4, 4, 2048, 2048, 128, 0, 128, 2),
    (torch.float16, 1, 4, 4, 2049, 2049, 128, 0, 128, 2),
    
    # 不同的 head_size
    (torch.float16, 1, 4, 4, 256, 256, 64, 0, 128, 2),
    (torch.float16, 1, 4, 4, 256, 256, 128, 0, 128, 2),
    (torch.float16, 1, 4, 4, 256, 256, 256, 0, 128, 2),
    
    # 不同的缓存模式
    (torch.float16, 1, 4, 4, 256, 256, 128, 0, 128, 2),  # 常规模式
    (torch.float16, 1, 4, 4, 256, 256, 128, 1, 128, 2),  # 块缓存模式
    
    # 因果掩码设置
    (torch.float16, 1, 4, 4, 256, 256, 128, 0, 128, 2),  # 非因果
    (torch.float16, 1, 4, 4, 256, 256, 128, 0, 128, 2),   # 因果
    
    # 现有测试用例
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 0, 128, 1),
    (torch.float16, 1, 8, 2, 512, 512, 64, 0, 128, 2),
    (torch.float16, 2, 12, 3, 256, 256, 128, 1, 128, 2),
]

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, mask_type", test_cases)
def test_fa_custom_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, mask_type):
    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0
    block_size = 128
    num_blocks = 64
    query = (q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size, q_seqlen, num_heads, head_size)).to(data_type).npu()
    key_cache = None
    value_cache = None
    block_tables = []
    if cache_mode == 1:
        key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size)).to(data_type).npu()
        value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size)).to(data_type).npu()
        max_num_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
        for i in range(batch_size):
            block_table = [
                max_num_blocks_per_seq * i + j
                for j in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int32).npu()
    else:
        key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
        value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
        block_tables = None
    kv_seqlen_list = [kv_seqlen] * batch_size
    scale = 1.0 / (head_size ** 0.5)
    window_size_left = -1
    window_size_right = -1
    is_rotary_interleaved = False
    softcap = 0
    num_splits = 0
    kv_seqlen_list = torch.tensor(kv_seqlen_list, dtype=torch.int64).cpu()
    rotary_cos = None
    rotary_sin = None
    cache_batch_idx = None
    leftpad_k = None
    alibi_slopes = None
    if mask_type == 2:
        # Generate alibi slopes: num_heads
        alibi_slopes = torch.rand(num_heads, device=query.device, dtype=torch.float32) * 0.3
    out_out, softmax_lse = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        None,
        None,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=kv_seqlen_list,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
        block_table=block_tables,
        causal=mask_type,
        window_size=[window_size_left, window_size_right],
        rotary_interleaved=is_rotary_interleaved,
        alibi_slopes=alibi_slopes,
        num_splits=num_splits,
        return_softmax_lse=True
    )
    golden_out = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
    golden_lseL = torch.empty((batch_size, q_seqlen, num_heads), dtype=torch.float32)
    atten_mask = None
    if  mask_type:
        atten_mask = torch.triu(torch.ones(q_seqlen, kv_seqlen), diagonal=1).bool()
    for i in range(batch_size):
        key_cache_per_batch = None
        value_cache_per_batch = None
        if cache_mode == 1:
            keys = []
            values = []
            block_table = block_tables.cpu()[i]
            for j in range(kv_seqlen):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = key_cache.detach().cpu()[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size)
                keys.append(k)
                v = value_cache.detach().cpu()[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size)
                values.append(v)
            key_cache_per_batch = torch.stack(keys, dim=0)
            value_cache_per_batch = torch.stack(values, dim=0)
        else:
            key_cache_per_batch = key_cache.detach().cpu()[i]
            value_cache_per_batch = value_cache.detach().cpu()[i]
        query_cpu = query.detach().cpu()[i]
        alibi_slopes_cpu = alibi_slopes.cpu() if alibi_slopes is not None else None
        leftpad_k_cpu = leftpad_k.cpu() if leftpad_k is not None else None
        if mask_type:
            output, golden_lse = ref_flash_attention(query_cpu, key_cache_per_batch, value_cache_per_batch, scale, atten_mask, data_type, alibi_slopes_cpu, leftpad_k_cpu)
        else:
            output, golden_lse = ref_flash_attention(query_cpu, key_cache_per_batch, value_cache_per_batch, scale, None, data_type, alibi_slopes_cpu, leftpad_k_cpu)
        out = output.reshape(q_seqlen, num_heads, head_size)
        golden_out[i:i+1] = out
        golden_lseL[i:i+1] = torch.transpose(golden_lse.reshape(num_heads, q_seqlen), 0, 1)
    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(out_out.cpu(), golden_out.cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(softmax_lse.cpu(), golden_lseL.cpu(), rtol=rtol, atol=atol)