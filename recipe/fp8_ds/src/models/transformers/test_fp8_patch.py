import random
import torch
from typing import Tuple
import triton
import pandas as pd
from .fp8_patch import DeepLinear, deep_matmul, per_token_cast_to_fp8, per_block_cast_to_fp8
from copy import deepcopy
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'


device = 'cuda'
dtype = torch.bfloat16
m, n, k = 4096 * 4, 4096, 4096
x1 = torch.randn(m, k, dtype=dtype, device=device)
x1.requires_grad_(True)
x2 = deepcopy(x1)
fc1 = torch.nn.Linear(k,n, bias=False, dtype=dtype, device=device)
fc2 = DeepLinear(k,n, bias=False, dtype=dtype, device=device)
fc2.weight.data.copy_(fc1.weight.data)

y1 = fc1(x1)
y2 = fc2(x2)
dy = torch.randn_like(y1)
y1.backward(dy)
y2.backward(dy)
print(y1)
print(y2)
print(x1.grad)
print(x2.grad)

print(triton.testing.do_bench(lambda: fc1(x1)))
print(triton.testing.do_bench(lambda: fc2(x2)))
y1 = fc1(x1)
y2 = fc2(x2)
dy = torch.randn_like(y1)
print(triton.testing.do_bench(lambda: y1.backward(dy, retain_graph=True)))
print(triton.testing.do_bench(lambda: y2.backward(dy, retain_graph=True)))