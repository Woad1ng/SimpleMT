#!/usr/bin/env bash

set -e

res_path=exp_SimpleMT/results


models="qwen2 nllb sm4tv2 muss"
methods="aoa wordfreq"


eval_script=src/evaluate.py



for model in $models; do
  for method in $methods; do
    result_file="$res_path/result_${model}_${method}.json"

    output_field="${model}_asset.test.zh.orig_${method}"
    echo "Evaluating $result_file, output_field: $output_field"
    python $eval_script \
      --input_file $result_file \
      --output_field $output_field \
    echo ""
  done
done