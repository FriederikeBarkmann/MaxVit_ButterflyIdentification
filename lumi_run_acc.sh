#!/bin/bash

# Make Python tools available
$WITH_CONDA

# Training settings
MODELS=(resnet50 resnet101 resnet152 \
        wide_resnet50_2 wide_resnet101_2 \
        resnext101_32x8d resnext50_32x4d resnext101_64x4d \
        regnet_y_16gf regnet_y_32gf regnet_y_128gf regnet_y_3_2gf \
        regnet_x_8gf regnet_x_16gf regnet_x_32gf regnet_x_3_2gf \
        densenet121 densenet169 densenet201 densenet161 \
        vgg19 vgg19_bn vgg16_bn \
        efficientnet_v2_s efficientnet_v2_m efficientnet_v2_l \
        vit_b_16 vit_b_32 vit_l_32 vit_l_16 vit_h_14 \
        swin_v2_t swin_v2_s swin_v2_b \
        maxvit_t \
        convnext_tiny convnext_small convnext_base convnext_large)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
MODEL="maxvit_t"
MODEL_PATH="$RESULTS_DIR/maxvit_t/10341785_2025-04-13_17-52/pytorch_model.bin"
#MODEL="convnext_large"
#MODEL_PATH="$RESULTS_DIR/convnext_large/10572172_2025-04-29_12-28/pytorch_model.bin"
BATCH_SIZE=8  # Batch size *per GPU*
EPOCHS=1
BASE_LR=5e-7  # Scales linearly with devices in code
WEIGHT_DECAY=1e-6
OPTIMIZER="Adam"  # SGD, ASGD, RMSprop, Adam, AdamW, Adadelta, Adagrad
SCHEDULER="ReduceLROnPlateau"  # ConstantLR, StepLR, ReduceLROnPlateau, CosineAnnealingLR
OVERSAMPLING=1  # 0 for False (using class weights) or 1 for True
CHECKPOINTING=0  # 0 for False or 1 for True (checkoint after each epoch)


# Keep track of some settings
if [[ ${SLURM_PROCID} -eq 0 ]]; then
    echo
    echo "Model: $MODEL"
    echo "Number of GPUs: $(($SLURM_JOB_NUM_NODES*$SLURM_GPUS_PER_NODE))"
    echo "Batch size per GPU: $BATCH_SIZE"
    #echo "Oversampling: $OVERSAMPLING"
    echo "Base learning rate (scales by number of GPUs): $BASE_LR"
    echo "Optimizer: $OPTIMIZER"
    echo "Scheduler: $SCHEDULER"
fi


# DPP with HF Accelerate launched with accelerate launch
accelerate launch \
    --multi_gpu \
    --same_network \
    --machine_rank=$SLURM_PROCID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_machines=$SLURM_JOB_NUM_NODES \
    --num_processes=$(($SLURM_JOB_NUM_NODES*$SLURM_GPUS_PER_NODE)) \
    --num_cpu_threads_per_process=$(($SLURM_CPUS_PER_TASK/$SLURM_GPUS_PER_NODE)) \
    --rdzv_backend=static \
    --mixed_precision="no" \
    --dynamo_backend="no" \
    /workdir/run_acc.py \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --base-lr $BASE_LR \
    --weight-decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --epochs $EPOCHS \
    --oversampling $OVERSAMPLING \
    --checkpointing $CHECKPOINTING \
    --data-dir $DATA_DIR \
    --results-dir $RESULTS_DIR \
    --model-path $MODEL_PATH \
    #--checkpoint $RESULTS_DIR/maxvit_t/10241322_2025-04-07_12-07/checkpoint_99

