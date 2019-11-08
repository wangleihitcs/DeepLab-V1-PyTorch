#!/bin/bash
# test and evaluate mIOU
models=`ls -t -r ./exp/model_last_*.pth`
for model_path in $models
do
    # echo $model_path
    CUDA_VISIBLE_DEVICES=0 python main.py --type=test --model_path_test=$model_path
    python evaluate.py
done