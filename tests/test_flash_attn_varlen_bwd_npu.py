import sys
import os
import torch
import torch_npu
import pytest
import flash_attn_2_cuda
import numpy as np
from flash_attn import flash_attn_varlen_func_backward

torch.npu.set_device(1)
np.random.seed(3)
torch.manual_seed(3)


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype = torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu


def test_tnd_bwd_npu(nheads, nheads_k, headdim, list_seq):
    g = nheads / nheads_k
    scale = 1 / (headdim ** 0.5)
    seqlens_list_q = np.array(list_seq)
    seqlens_list_k = np.array(list_seq)
    B = len(list_seq)

    keep_prob = 1.0
    max_seqlen_q = np.max(seqlens_list_q)
    max_seqlen_k = np.max(seqlens_list_k)
    cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
    cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
    total_q = seqlens_list_q.sum()
    total_k = seqlens_list_k.sum()
    cu_seq_len_list = cu_seqlens_q[1:].cpu().numpy().tolist()
    cu_seq_kvlen_list = cu_seqlens_k[1:].cpu().numpy().tolist()
    print("total_q: ", total_q)
    print("total_k: ", total_k)
    print("cu_seq_len_list is ", cu_seq_len_list)
    print("cu_seq_kvlen_list is ", cu_seq_kvlen_list)
    
    pttype = torch.float16
    limit = 2
    q = limit * (torch.rand([total_q, nheads, headdim]) - 0.5).to(pttype)
    k = limit * (torch.rand([total_k, nheads_k, headdim]) - 0.5).to(pttype)
    v = limit * (torch.rand([total_k, nheads_k, headdim]) - 0.5).to(pttype)
    dout = limit * (torch.rand([total_q, nheads, headdim]) - 0.5).to(pttype)

    # q = torch.ones_like(q)
    # k = torch.ones_like(k)
    # v = torch.ones_like(v)
    # dout = torch.ones_like(dout)
    print("q.shape ", q.shape)
    print("k.shape ", k.shape)
    print("v.shape ", v.shape)
    print("dout.shape ", dout.shape)

    
    # npu attention mask args
    pre_tocken = 65536
    next_tocken = 0
    sparse_mode = 3

    # gpu attention mask args
    causal_switch = True
    window_left = 65536
    window_right = 0

    # call npu_fusion_attention golden
    q = q.npu()
    k = k.npu()
    v = v.npu()
    dout = dout.npu()
    atten_mask_npu = (torch.triu(torch.ones([2048, 2048]), diagonal=1)).to(torch.bool).npu()

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    torch.npu.synchronize()
    npu_rst = torch_npu.npu_fusion_attention(
            q, k, v, nheads,
            pse=None,
            padding_mask=None,
            atten_mask=atten_mask_npu,
            scale=scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seq_len_list),
            actual_seq_kvlen=tuple(cu_seq_kvlen_list),
            pre_tockens=pre_tocken,
            next_tockens=next_tocken,
            inner_precise=0,
            sparse_mode=sparse_mode,
            prefix=None)
    out_npu = npu_rst[0]
    x_max_npu = npu_rst[1]
    x_sum_npu = npu_rst[2]
    torch.npu.synchronize()
    out_npu.backward(dout)
    dq_golden_npu = q.grad
    dk_golden_npu = k.grad
    dv_golden_npu = v.grad
    torch.npu.synchronize()

    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int64).cpu()
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int64).cpu()

    # call tridao npu interface
    print("cu_seqlens_q is ", cu_seqlens_q)
    print("cu_seqlens_k is ", cu_seqlens_k)

    dq_tridao, dk_tridao, dv_tridao = flash_attn_varlen_func_backward(
        dout,
        q,
        k,
        v,
        out_npu,
        # cpu_softmax_log_sum_nt.float(),
        x_max_npu,
        x_sum_npu,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=(1 - keep_prob),
        softmax_scale=scale,
        causal=causal_switch,
        window_size=(window_left, window_right),
        alibi_slopes=None
    )
    torch.npu.synchronize()

    from precision_compare import data_compare
    data_compare(dq_tridao.cpu().float().numpy(), dq_golden_npu.cpu().float().numpy())
    data_compare(dk_tridao.cpu().float().numpy(), dk_golden_npu.cpu().float().numpy())
    data_compare(dv_tridao.cpu().float().numpy(), dv_golden_npu.cpu().float().numpy())



test_tnd_bwd_npu(1, 1, 128, [512])
