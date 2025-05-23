#! /bin/sh
## Run this script using command "sh ~.sh"

CUDA_VISIBLE_DEVICES=1 python ../test_proposals.py --model siamese_bn --finetune --opt test_w32 --epoch 9