#! /bin/sh
## Run this script using command "sh ~.sh"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../test_proposals_split.py --model siamese_stn_late_bn --finetune --opt test_w32 --epoch 9 --index 0