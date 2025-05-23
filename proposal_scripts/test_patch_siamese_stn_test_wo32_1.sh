#! /bin/sh
## Run this script using command "sh ~.sh"

CUDA_VISIBLE_DEVICES=3 python ../test_proposals_split.py --model siamese_stn_late_bn --finetune --opt test_wo32 --epoch 11 --index 1
