#!/bin/bash
#SBATCH -J simnpo_repro_test       # Job name
#SBATCH -o slurm-outputs/%j.out    # Output file (%j = job ID)
#SBATCH -t 2:00:00                # Wall time (24 hours)
#SBATCH -N 8                       # Number of nodes
#SBATCH --ntasks-per-node 1      # total number of tasks per node
#SBATCH -p gh-dev                      # GPU partition
#SBATCH -A ASC25010                # Project allocation

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

module load cuda/12.6 nccl/12.4 nvidia_math
source /work/10010/jacoblblock/vista/miniconda3/etc/profile.d/conda.sh
conda activate llm-unlearn
source /work/10010/jacoblblock/vista/llm-unlearning/scripts/set_cache_and_unset.sh

export PYTHONUNBUFFERED=1

srun torchrun --nnodes 8 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 forget.py --config-name=forget.yaml split=forget05 npo_coeff=0.1375 beta=2.5
