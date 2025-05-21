import json
import pandas as pd
from pathlib import Path

def convert_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    train_data = []
    cnt = 0
    for k, v in data.items():
        if k == 'Info':
            continue
        v['id'] = cnt
        cnt += 1

        v['reward_model'] = {
            "ground_truth": v['gold'],
            'style': 'rule-lighteval/MATH_v2'
        }

        v['data_source'] = 'math_dapo'

        v['ability']  = 'MATH'
        v['prompt'] = [{
            'content': v['problem'][0]['prompt'],
            'role': 'user'
        }]
        v['extra_info'] = {
            'uid': k
        }
        
        train_data.append(v)

    df = pd.DataFrame(train_data)
    df.to_parquet(str(path.with_suffix('.parquet')))


if __name__ == '__main__':
    root_path = Path('/home/projects/polyullm/congkai/data/verl/data/infiq')
    for path in root_path.glob("*/diff_result_uid_vanilla_only.json"):
        convert_data(path)





