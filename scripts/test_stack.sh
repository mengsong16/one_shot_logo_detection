#! /bin/sh
## Run this script using command "sh ~.sh"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python ../test.py --model vgg16_stack_bn --finetune --classid --start_epoch 0 --end_epoch 14 --opt test_w32
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python ../test.py --model vgg16_stack_bn --finetune --classid --start_epoch 0 --end_epoch 14 --opt test_wo32
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python ../test.py --model vgg16_stack_bn --finetune --classid --start_epoch 0 --end_epoch 14 --opt test_seen