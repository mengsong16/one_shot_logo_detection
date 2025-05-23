#! /bin/sh


## Run this script using command "sh ~.sh"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../test.py --model siamese_bn --finetune --classid --hard_mining  --negative_only --start_epoch 0 --end_epoch 14  