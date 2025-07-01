#!/usr/bin/env bash

set -e  # exit if error

# prepare folders
exp_path=exp_SimpleMT
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path


models="qwen2 nllb sm4tv2 muss"
methods="aoa wordfreq"


dataset=./data/asset/asset.test.zh.orig


num_beams=10
src_lang=cmn
tgt_lang=eng

echo "Evaluate models with different methods:"

for model in $models; do
  for method in $methods; do
    model_dir="./model/$model"  
    output_file="$res_path/result_${model}_${method}.json"
    echo "Evaluating $model with $method ..."
    python SimpleMT.py \
      --model_dir $model_dir \
      --dataset_dir $dataset \
      --output_path $output_file \
      --num_beams $num_beams \
      --logits_processor $method \
      --src_lang $src_lang \
      --tgt_lang $tgt_lang
  done
done