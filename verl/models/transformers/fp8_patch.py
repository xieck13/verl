import torch
import triton
import triton.language as tl
from typing import Tuple
import deep_gemm
import re
import types


# @triton.autotune(
#         [triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                 for bsm in [16, 32]
#                 for ns in [1, 2, 4]
#                 for nw in [1, 2, 4, 8]
#                 ], key=['M', 'N'])
@triton.jit
def _per_token_cast_to_fp8_kernel(X, Y, S,
                           stride_xm, stride_xn,
                           stride_ym, stride_yn,
                           stride_sk, stride_sm,
                           M, N, K, MAX,
                           BLOCK_M: tl.constexpr=32, BLOCK_N: tl.constexpr=128,
                            ):
    off_m = tl.cast(tl.program_id(axis=0), tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    pid_k = tl.cast(tl.program_id(axis=1), tl.int64)
    off_n = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = off_m < M

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask[:, None], other=0.).to(tl.float32)
    x_max = tl.clamp(tl.max(tl.abs(x), -1), min=0, max=1e4) + 0.000001
    scale = x_max / MAX
    y = x / scale[:, None]
    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y, mask=mask[:, None])
    tl.store(S + off_m * stride_sm + pid_k * stride_sk, scale, mask=mask)


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_N = 128
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    K = N // BLOCK_N
    aligin_m = triton.cdiv(M, 8) * 8
    s = torch.empty(triton.cdiv(N, BLOCK_N), aligin_m, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), K)
    if x.is_contiguous():
        kwargs = {'BLOCK_M': 32, 'BLOCK_N': BLOCK_N, 'num_warps':8, 'num_stages':2}
    else:
        kwargs = {'BLOCK_M': 32, 'BLOCK_N': BLOCK_N, 'num_warps':1, 'num_stages':4}
    _per_token_cast_to_fp8_kernel[grid](x, y, s,
                        *x.stride(),
                        *y.stride(),
                        *s.stride(),
                        M, N, K, torch.finfo(torch.float8_e4m3fn).max,
                        **kwargs,
                        )
    return y, s.transpose(0, 1)

# @triton.autotune(
#         [triton.Config({}, num_stages=ns, num_warps=nw)
#                 for ns in [1, 2, 3, 4]
#                 for nw in [1, 2, 4, 8]
#                 ], key=['M', 'N']
# )
@triton.jit
def _per_block_cast_to_fp8_kernel(X, Y, S,
                           stride_xm, stride_xn,
                           stride_ym, stride_yn,
                           stride_sm, stride_sn,
                           M, N, MAX,
                           BLOCK_M: tl.constexpr=128, BLOCK_N: tl.constexpr=128,
                            ):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.).to(tl.float32)
    x_max = tl.clamp(tl.max(tl.abs(x)), min=0, max=1e4) + 0.000001
    scale = x_max / MAX
    y = x / scale
    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y, mask=mask)
    tl.store(S + pid_m * stride_sm + pid_n * stride_sn, scale)

# @triton.autotune(
#         [triton.Config({}, num_stages=ns, num_warps=nw)
#                 for ns in [1, 2, 3, 4]
#                 for nw in [1, 2, 4, 8]
#                 ], key=['M', 'N']
# )
# @triton.jit
# def _per_block_cast_to_fp8_kernel(X, Y, S,
#                            stride_xm, stride_xn,
#                            stride_ym, stride_yn,
#                            stride_sm, stride_sn,
#                            M, N, MAX,
#                            BLOCK_M: tl.constexpr=128, BLOCK_N: tl.constexpr=128,
#                             ):
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
#     start_m = pid_m * BLOCK_M
#     start_n = pid_n * BLOCK_M

#     x_ptrs = tl.make_block_ptr(X, (M, N), (stride_xm, stride_xn), (start_m, start_n), (BLOCK_M, BLOCK_N), (1, 0))
#     y_ptrs = tl.make_block_ptr(Y, (M, N), (stride_ym, stride_yn), (start_m, start_n), (BLOCK_M, BLOCK_N), (1, 0))
#     x = tl.load(x_ptrs, boundary_check=(0, 1)).to(tl.float32)
#     x_max = tl.clamp(tl.max(tl.abs(x)), min=0, max=1e4) + 0.000001
#     scale = x_max / MAX
#     y = x / scale
#     tl.store(y_ptrs, y.to(y_ptrs.dtype.element_ty), boundary_check=(0, 1))
#     tl.store(S + pid_m * stride_sm + pid_n * stride_sn, scale)

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_M = 128
    BLOCK_N = 128
    M, N = x.shape
    A, B = triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N)
    y = torch.empty(A * BLOCK_M, B * BLOCK_N,
                    device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(A, B, dtype=torch.float32, device=x.device)
    grid = (A, B)
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'num_warps':8, 'num_stages':3}
    _per_block_cast_to_fp8_kernel[grid](x, y, s,
                        *x.stride(),
                        *y.stride(),
                        *s.stride(),
                        M, N, torch.finfo(torch.float8_e4m3fn).max,
                        **kwargs,
                        )
    return y, s

def pad_8(tensor):
    m = tensor.shape[0]
    if m % 8 == 0:
        return tensor
    num_pad = 8 - m % 8
    return torch.nn.functional.pad(tensor, (0, 0, 0, num_pad), "constant", 0)

def pad_256(tensor):
    k = tensor.shape[1]
    if k % 256 == 0:
        return tensor
    num_pad = 256- k % 256
    return torch.nn.functional.pad(tensor, (0, num_pad, 0, 0), "constant", 0)


def deep_matmul(a, b, out=None):
    """
    fp8 matmul, it contain quant and matmul

    Args:
        a (torch.Tensor): shape : [m, k], dtype: torch.bfloat16, don't must be contiguous
        a (torch.Tensor): shape : [n, k], dtype: torch.bfloat16, don't must be contiguous

    Return:
        out (torch.Tensor):  shape : [m, n], dtype: torch.bfloat16
    """
    assert a.dim() == 2 and a.dtype == torch.bfloat16
    assert b.dim() == 2 and b.dtype == torch.bfloat16
    m, k = a.shape
    n, k2 = b.shape
    assert k == k2 and k % 128 == 0
    a = pad_8(a)
    a_fp8 = per_token_cast_to_fp8(a)
    b_fp8 = per_block_cast_to_fp8(b)
    if out is None:
        out = torch.empty(a.size(0), n, device=a.device, dtype=torch.bfloat16)
    deep_gemm.gemm_fp8_fp8_bf16_nt(a_fp8, b_fp8, out)
    return out[:m]

# Te中用的，不用管
def deep_matmul_pad256(a, b, out=None):
    # 注意，a是[m,k]，b是[n,k]，不要求连续
    assert a.dim() == 2 and a.dtype == torch.bfloat16
    assert b.dim() == 2 and b.dtype == torch.bfloat16
    m, k = a.shape
    n, k2 = b.shape
    assert k == k2
    assert m % 8 == 0 and n % 8 == 0
    a = pad_256(a)
    b = pad_256(b)
    a_fp8 = per_token_cast_to_fp8(a)
    b_fp8 = per_block_cast_to_fp8(b)
    # print(a_fp8[0].shape, a_fp8[1].shape, y_fp8[0].shape, y_fp8[1].shape)
    if out is None:
        out = torch.empty(a.size(0), n, device=a.device, dtype=torch.bfloat16)
    deep_gemm.gemm_fp8_fp8_bf16_nt(a_fp8, b_fp8, out)
    return out





# @triton.autotune(
#         [triton.Config({'BLOCK_SIZE': bs}, num_stages=ns, num_warps=nw)
#                 for bs in [2048, 4096, 8192]
#                 for ns in [1, 2, 4]
#                 for nw in [4, 8]
#                 ], key=['N'])
@triton.jit
def _copy_kernel(X, Y, Ind, stride,
                N, BLOCK_SIZE:tl.constexpr):
    x_row_id = tl.program_id(0)
    y_row_id = tl.load(Ind + x_row_id)

    X += x_row_id * stride
    Y += y_row_id * stride

    for start in range(0, N, BLOCK_SIZE):
        cols = start+tl.arange(0, BLOCK_SIZE)
        y = tl.load(Y + cols, mask=cols<N, other=0.)
        tl.store(X + cols, y, mask=cols<N)

def tensor_copy(x, y, indices):
    kwargs = {'BLOCK_SIZE': 2048, 'num_warps':8, 'num_stages':1}
    _copy_kernel[(x.size(0), )](x, y, indices,
                                x.stride(0),
                                x.size(1),
                                **kwargs
                                # BLOCK_SIZE
                                )


# Te中用的，不用管
def deep_group_matmul(a_list, b_list, m_splits, out=None):
    a = a_list[0]
    b = b_list[0]
    assert a.dim() == 2 and a.dtype == torch.bfloat16
    assert b.dim() == 2 and b.dtype == torch.bfloat16
    m, k = a.shape
    n, k2 = b.shape
    assert k == k2 and k % 128 == 0, f"k: {k}, k2: {k2}"
    num_groups = len(m_splits)

    b_fp8 = (torch.empty(num_groups, n, k, dtype=torch.float8_e4m3fn, device=a.device),
            torch.empty((num_groups, (n + 127) // 128, k // 128), device=a.device, dtype=torch.float))
    if out is None:
        out = torch.empty(sum(m_splits), n, device=a.device, dtype=torch.bfloat16)

    need_pad = not all([i % 8 == 0 for i in m_splits])
    if need_pad:
        a_list = [pad_8(tensor) for tensor in a_list]
        pad_m_splits = [tensor.size(0) for tensor in a_list]
        a_fp8 = per_token_cast_to_fp8(torch.cat(a_list, axis=0))
        indices = []
        for i in range(len(b_list)):
            b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b_list[i])
            indices.extend([i] * pad_m_splits[i])
        indices = torch.tensor(indices, dtype=torch.int32, device=a.device)
        pad_out = torch.empty(a_fp8[0].size(0), n, device=a.device, dtype=torch.bfloat16)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(a_fp8, b_fp8, pad_out, indices)
        pad_indices = torch.arange(0, sum(m_splits), device=a.device, dtype=torch.int32)
        start = 0
        for i in range(len(m_splits)-1):
            start += m_splits[i]
            pad_indices[start:] += (pad_m_splits[i] - m_splits[i])
        tensor_copy(out, pad_out, pad_indices)
    else:
        a_fp8 = per_token_cast_to_fp8(torch.cat(a_list, axis=0))
        indices = []
        for i in range(len(b_list)):
            b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b_list[i])
            indices.extend([i] * m_splits[i])
        indices = torch.tensor(indices, dtype=torch.int32, device=a.device)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(a_fp8, b_fp8, out, indices)
    return out



class _DeepLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        ctx.inp_shape = inp.shape
        inp = inp.view(-1, inp.size(-1))
        ctx.save_for_backward(inp, weight, bias)
        out = deep_matmul(inp, weight)
        if bias is not None:
            out += bias
        return out.view(*ctx.inp_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inp, weight, bias = ctx.saved_tensors
        grad_outputs = grad_outputs.view(-1, grad_outputs.size(-1))
        dbias = None
        dinp, dweight, dbias = None, None, None

        dweight = deep_matmul(grad_outputs.T, inp.T)
        if inp.requires_grad:
            dinp = deep_matmul(grad_outputs, weight.T).view(ctx.inp_shape)

        if bias is not None:
            dbias = grad_outputs.sum(0)
        return dinp, dweight, dbias


class DeepLinear(torch.nn.Linear):
    def forward(self, input):
        return _DeepLinear.apply(input, self.weight, self.bias)

def fp8_forward(self, input):
    assert input.dtype == torch.bfloat16
    assert self.weight.dtype == torch.bfloat16
    return _DeepLinear.apply(input, self.weight, self.bias)

def apply_fp8_patch(module, pattern=['proj'], prefix=None):
    '''
    apply fp8 for the specific Linear

    Args:
        module (torch.nn.Module): the input is the Model Like Qwen or Llama.
        pattern (List[str]): Which Linear will be apply fp8. If the module name match one of pattern,
                            the module will apply fp8. It can like "[proj]" or [q_proj, up_proj, gate_proj].
                            Warning: Don't use it for lm_head
        prefix: set to None at the begining, you can  print the full_name to verfiy the accurate.
    '''
    if module is None:
        return
    for name, child in module.named_children():
        full_name = name if prefix is None else prefix + '.' + name
        if isinstance(child, torch.nn.Linear):
            for p in pattern:
                if re.search(p, name):
                    # print(full_name)
                    child.forward = types.MethodType(fp8_forward, child)
                    break
        apply_fp8_patch(child, pattern, full_name)


