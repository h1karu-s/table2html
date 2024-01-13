# table2html

### swin-bart
```
    CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python3 -m torch.distributed.launch --nproc_per_node 4 ./src/train_swin-bart.py \
        --input_file_dir <your dataset dir> \
        --input_file <train file(pickle)> \
        --input_val_file <valid file (pickle)> \
        --input_test_file <test file (pickle )> \
        --output {output_dir} \
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
        --seed 40 \
        --max_length 1024 \
        --use_amp \
        --use_ddp
```

### layoutlmv3-bart
```
    CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python3 -m torch.distributed.launch --nproc_per_node 4 ./src/train_layoutlmv3-bart.py \
        --input_file_dir <your dataset dir> \
        --input_file <train file(pickle)> \
        --input_val_file <valid file (pickle)> \
        --input_test_file <test file (pickle )> \
        --output {output_dir} \
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
```
