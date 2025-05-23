#! /bin/sh


## Run this script using command "sh ~.sh"
CUDA_VISIBLE_DEVICES=7 python ../test_embedding.py --model siamese_contrastive_bn --finetune --start_epoch 0 --end_epoch 9 --opt test_seen