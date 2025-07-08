# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

export HOME=/home/projects/polyullm/congkai/data/verl
export WANDB_MODE=offline
export RAY_DEDUP_LOGS_ALLOW_REGEX='xieck13'
export NCCL_TIMEOUT=7200000  # 2小时
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

DATA_DIR="$HOME/data/geo3k_multiturn_w_tool"
# WANDB_API_KEY=XXX
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NO_PROXY="localhost,127.0.0.1,klb-dgx-*"
export no_proxy="localhost,127.0.0.1,klb-dgx-*"
export WANDB_DIR="/home/projects/polyullm/congkai/data/verl/wandb"
export RAY_TMPDIR="/zju_0038/tmp"

rm -rf /lustre/projects/polyullm/congkai/verl_will_v3.0/tmp/*

RAY_OVERRIDE_JOB_RUNTIME_ENV=1 RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . \
    --runtime-env-json='{
        "env_vars": {
            "HOME": "/home/projects/polyullm/congkai/data/verl",
            "WANDB_MODE": "offline",
            "WANDB_DIR": "/home/projects/polyullm/congkai/data/verl/wandb",
            "RAY_DEDUP_LOGS_ALLOW_REGEX": "xieck13",
            "NCCL_TIMEOUT": "7200000",
            "RAY_TMPDIR": "/zju_0038/tmp",
            "NO_PROXY": "localhost,127.0.0.1,klb-dgx-*",
            "no_proxy": "localhost,127.0.0.1,klb-dgx-*"
        }
    }' \
    -- python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='geo3k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=512 \
    data.max_prompt_length=1536 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="/lustre/projects/polyullm/congkai/models/ta_merge/Qwen2.5-VL-7B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="geo3k_multiturn" \
    trainer.experiment_name="Qwen2.5-VL-7B-Tool-ta_merge" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml" \
    trainer.total_epochs=2 $@


cp -r /tmp/ray /lustre/projects/polyullm/congkai/verl_will_v3.0/tmp
