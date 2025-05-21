import pandas as pd

configs = [
    {
        'model': '/lustre/projects/polyullm/models/Qwen2.5-1.5B-Instruct',
        'data_path': '/home/projects/polyullm/congkai/data/verl/data/infiq/Qwen2.5-1.5B-Instruct-r32-a64-lr5e-06-11611-1epoch_vs_Qwen2.5-1.5B-Instruct-AWQ-r32-a64-lr5e-06-11611-1epoch/diff_result_uid_vanilla_only.parquet',
        'exp_name': 'DAPO-Qwen2.5-1.5B-Instruct-AWQ-fp8',
        'tensor_parallel': 1,
        'nnodes': 1,
    },
    {
        'model': '/lustre/projects/polyullm/models/Qwen2.5-1.5B-Instruct',
        'data_path': 'xxx',
        'exp_name': 'xxx',
    },
]


