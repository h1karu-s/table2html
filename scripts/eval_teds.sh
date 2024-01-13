
model_name=swin-bart
dataset_name=fintabnet


for seed in 40
do
    python3 ./src/eval_teds.py \
        --pred_path ./data/train/${dataset_name}/${model_name}/${seed}/best_model/pred_test.json \
        --gt_path ./data/train/${dataset_name}/${model_name}/${seed}/best_model/gt_test.json \
        --output_dir ./data/train/${dataset_name}/${model_name}/${seed}/best_model
done