#!/bin/bash


# LUMI data and results directories to bind
export DATA_DIR=/flash/$SLURM_JOB_ACCOUNT/$USER/datasets
export RESULTS_DIR=/scratch/$SLURM_JOB_ACCOUNT/$USER/butterflies/results
mkdir -p $RESULTS_DIR

# Some PyTorch container with the necessary packages
export SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif


# Execute _script_ in container to avoid "address already in use" error
singularity exec \
    -B $DATA_DIR \
    -B $RESULTS_DIR \
    -B $PWD:/workdir \
    $SIF /workdir/$SCRIPT

