#!/bin/bash
#SBATCH --job-name=verl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1024G
#SBATCH --partition=AISS2025031801
#SBATCH --time=192:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=/home/projects/polyullm/congkai/logs/verl-%j.out
#SBATCH --error=/home/projects/polyullm/congkai/logs/verl-%j.err

# set -x

# replace these information with your own
verl_workdir=/lustre/projects/polyullm/congkai/verl_will_v3.0
container_image=/lustre/projects/polyullm/container/verl-sglang+0503.sqsh
container_name=verl-sglang+0503
container_mounts=/lustre/projects/polyullm:/lustre/projects/polyullm,/home/projects/polyullm:/home/projects/polyullm,/lustre/projects/polyullm/congkai/async_vlm_workspace_v2/zju_0038:/zju_0038


# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Start Ray head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

mkdir -p /home/projects/polyullm/tmp/$head_node


echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    --container-mounts=$container_mounts,/home/projects/polyullm/tmp/$head_node:/tmp \
    --container-image=$container_image \
    --container-workdir=$verl_workdir \
    --container-writable \
    --container-remap-root \
    bash -c "source /zju_0038/congkai/.python/verl-sglang/bin/activate && ray start --head --node-ip-address=$head_node_ip --port=$port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block" &

sleep 5

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    mkdir -p /home/projects/polyullm/tmp/$node_i
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        --container-name=$container_name \
        --container-mounts=$container_mounts,/home/projects/polyullm/tmp/$node_i:/tmp \
        --container-image=$container_image \
        --container-workdir=$verl_workdir \
        --container-writable \
        --container-remap-root \
        bash -c "source /zju_0038/congkai/.python/verl-sglang/bin/activate && ray start --address $ip_head --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block" &
    sleep 5
done

echo "Waiting for 60 seconds..."
sleep 60
echo "Starting training..."

SCRIPTS="
source /zju_0038/congkai/.python/verl-sglang/bin/activate && bash $@
"

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-image=$container_image \
    --container-workdir=$verl_workdir \
    --container-writable \
    --container-remap-root \
    bash -c "$SCRIPTS"


# Clean up Ray processes
cleanup() {
    echo "Shutting down Ray cluster..."
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        --container-name=$container_name \
        --container-mounts=$container_mounts \
        --container-image=$container_image \
        --container-writable \
        --container-remap-root \
        bash -c "c&& source /zju_0038/congkai/.python/verl-sglang/bin/activate && ray stop"

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        srun --overlap --nodes=1 --ntasks=1 -w "$node_i" \
            --container-name=$container_name \
            --container-mounts=$container_mounts \
            --container-image=$container_image \
            --container-writable \
            --container-remap-root \
            bash -c "source /zju_0038/congkai/.python/verl-sglang/bin/activate && ray stop"
    done
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT
