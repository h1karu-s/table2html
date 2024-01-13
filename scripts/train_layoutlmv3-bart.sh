#!/bin/bash
#SBATCH --output=./train_small_donut
# --master_port=25678
for seed in 40
do
    CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python3 -m torch.distributed.launch --nproc_per_node 4 ./src/train_layoutlmv3-bart.py \
        --input_file_dir /data/pubtabnet \
        --input_file complete_html_my_vocab_v1_tesseract-ocr_raw/train.pkl \
        --input_val_file complete_html_my_vocab_v1_tesseract-ocr_raw/val.pkl \
        --input_test_file complete_html_my_vocab_v1_tesseract-ocr_raw/val.pkl \
        --output ./data/train/pubtabnet/layoutlmv3-bart/${seed} \
        --ratio_train 0.9 \
        --train_batch_size 4 \
        --val_batch_size 8 \
        --train_epoch 15 \
        --train_accum_iter 14 \
        --store_names iter train_loss val_loss \
        --valid_ratio 0.3 \
        --train_clip_grad 1 \
        --learning_rate 1e-4 \
        --save_freq 4 \
        --check_grad_norm \
        --seed ${seed} \
        --num_encoder_layer 6 \
        --encoder_max_length 512 \
        --decoder_max_length 1024 \
        --use_amp \
        --use_ddp
done