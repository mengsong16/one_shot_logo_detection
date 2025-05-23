#! /bin/sh
## Run this script using command "sh ~.sh"

CUDA_VISIBLE_DEVICES=2 python ../test_proposals.py --model siamese_bn --finetune --opt test_wo32 --epoch 11
