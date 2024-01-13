for seed in 40
do
    CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python3 -m torch.distributed.launch --nproc_per_node 4 ./src/train_swin-bart.py \
        --input_file_dir /data/pubtabnet \
        --input_file complete_html_my_vocab_v1/train.pkl \
        --input_val_file complete_html_my_vocab_v1/val.pkl \
        --input_test_file complete_html_my_vocab_v1/val.pkl \
        --output ./data/train/pubtabnet/swin-bart/${seed} \
        --model_id  swin-bart \
        --ratio_train 0.9 \
        --train_batch_size 4 \
        --val_batch_size 8 \
        --train_epoch 20 \
        --train_accum_iter 12 \
        --input_height 448 \
        --input_width 896 \
        --window_size 7 \
        --encoder_depth 2 2 14 2 \
        --encoder_num_heads 4 8 16 32 \
        --store_names iter train_loss val_loss \
        --valid_ratio 0.3 \
        --train_clip_grad 1 \
        --learning_rate 1e-4 \
        --save_freq 5 \
        --check_grad_norm \
        --seed ${seed} \
        --max_length 1024 \
        --use_amp \
        --use_ddp
done