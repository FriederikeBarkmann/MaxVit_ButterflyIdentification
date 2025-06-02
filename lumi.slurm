#!/bin/bash

##SBATCH --test-only
#SBATCH --job-name=bfly_test  # Job name
#SBATCH --account=project_465001276  # Project for billing

# Mail notifications are not configured on LUMI

#SBATCH -D /users/lindnera/eurocc/projects/butterflies/andi
#SBATCH --output=/users/lindnera/slurm_out/%A_%a_%x_%j.out
#SBATCH --error=/users/lindnera/slurm_out/%A_%a_%x_%j.err

##SBATCH --partition=standard-g  # Standard full-node GPU partition
#SBATCH --partition=small-g # Fewer nodes but 3 days possible
##SBATCH --partition=dev-g  # Dev GPU partition for shorter tests
#SBATCH --exclusive  # Specify on development and small partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # or --ntasks if variable used for srun
##SBATCH --gpus-per-task=8  # Problems wih AMD GPUs and SLURM
#SBATCH --gpus-per-node=8  # or --gres=gpu:8
##SBATCH --gpus-per-node=mi250:8  # or --gres=gpu:mi250:8
#SBATCH --cpus-per-task=56  # 1 core per GPU reserved for the system

#SBATCH --mem-per-gpu=60G  # Specifiy on development partition

#SBATCH --time=0-15:00:00
##SBATCH --time=0-01:00:00

##SBATCH --array=35,36,37,38

# When using srun with multithreading
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Use for data loader

# Explicitly set the number of GPUs per node
export SLURM_GPUS_PER_TASK=$SLURM_GPUS_PER_NODE

date
echo

# Keep track of acquired resources
echo
echo "Account:" $SLURM_JOB_ACCOUNT
echo "Job ID:" $SLURM_JOB_ID ", Job name:" $SLURM_JOB_NAME
echo "Cluster:" $SLURM_CLUSTER_NAME
echo "Partition:" $SLURM_JOB_PARTITION
echo "QOS:" $SLURM_JOB_QOS
echo "Number of nodes:" $SLURM_JOB_NUM_NODES
echo "Nodes:" $SLURM_JOB_NODELIST
echo "Node:" $SLURMD_NODENAME
echo "Number of tasks:" $SLURM_NTASKS
echo "Tasks per node:" $SLURM_TASKS_PER_NODE
echo "CPUs per task:" $SLURM_CPUS_PER_TASK
echo "GPUs per task:" $SLURM_GPUS_PER_TASK
echo "CPUs per GPU:" $SLURM_CPUS_PER_GPU
echo "Memory per node:" $SLURM_MEM_PER_NODE
echo "Number of array tasks:" $SLURM_ARRAY_TASK_COUNT
echo 

echo "All SLURM variables:"
env | grep 'SLURM_'
echo


# Access to Slingshot network and file system for containers
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

# Set slinghsot interfaces to be used by RCCL and RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3


# Decide to train or tune
TASK=train

if [[ $TASK == "tune" ]]; then
    echo "Tuning with Ray"
    worker_list=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
    head_node=${worker_list[0]}
    export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    export ray_gcs_port=$(($RANDOM%(35000-20000+1)+20000))
    export RAY_ADDRESS="$head_node_ip:$ray_gcs_port"
    export SCRIPT=lumi_run_ray.sh
else
    echo "Training with Accelerate"
    export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
    export MASTER_PORT=23479
    export SCRIPT=lumi_run_acc.sh
fi

# Run script in system-provided container
srun lumi_run_container.sh

# Keep track of consumed resources
echo
sacct -j $SLURM_JOB_ID \
--format=jobname%15,jobid,state,elapsed,nnodes,ncpus,ntasks,cputime,maxrss,partition%20,nodelist
