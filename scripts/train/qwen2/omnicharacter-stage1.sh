#!/bin/bash
set -x

# wandb login
echo "DLC seemed world size(actually nodes): ${WORLD_SIZE}"
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export GPUS_PER_NODE=8
export BATCH_SIZE=1
export GRADIENT_ACCU_STEPS=1
export MASTER_PORT=29501
export CPUS_PER_TASK=16
export QUOTA=reserved

export ROOT_PATH=/home/zhanghaonan/code/acl_extension
export DATA_PATH=${ROOT_PATH}/data_root/train_merged_dialogues_final_reordered_with_audio_2.json
export SAVE_PATH=omnicharacter++_stage1_5e-4_no_audio_2epoches # no audio记得去train.py改speech_prob!
export BASE_LR=5e-4 # 5e-4感觉直接崩掉了 2e-7可以
export VIT_LR=2e-6

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MAX_RETRIES=3
WAIT_TIME=20
retry_count=0

command_to_run="torchrun --nproc_per_node $GPUS_PER_NODE --master_port=29501 openomni/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path ${ROOT_PATH}/pretrained/openomni/qwen2 \
--version llava_qwen_2 \
--data_path ${DATA_PATH} \
--image_folder ./datasets \
--speech_folder /home/zhanghaonan/code/acl_extension/data_root/audio_data/train \
--mm_projector_type mlp2x_gelu \
--freeze_backbone False \
--vision_tower ./checkpoints/openai/clip-vit-large-patch14-336 \
--tune_mm_mlp_adapter False \
--freeze_mm_mlp_adapter True \
--tune_speech_generator_only False \
--speech_encoder ${ROOT_PATH}/pretrained/whisper-large-v3/large-v3.pt \
--speech_generator ${ROOT_PATH}/pretrained/openomni/pretrained/speech_generator/speech_generator_ar_16k.pt \
--unfreeze_mm_vision_tower False \
--mm_vision_tower_lr ${VIT_LR} \
--speech_generator_lr ${VIT_LR} \
--mm_projector_lr ${VIT_LR} \
--image_aspect_ratio anyres \
--group_by_modality_length True \
--mm_vision_select_layer -2 \
--mm_vision_select_feature patch \
--mm_patch_merge_type spatial_unpad \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ./checkpoints/${SAVE_PATH} \
--num_train_epochs 2 \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 7500 \
--save_total_limit 20 \
--learning_rate ${BASE_LR} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 8196 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--speech_generator_type "ar" \
--run_name ${SAVE_PATH} \
--dataloader_drop_last True \
--report_to tensorboard | tee  train_${SAVE_PATH}.log"

eval $command_to_run
