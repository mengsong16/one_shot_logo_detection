#! /bin/sh


## Run this script using command "sh ~.sh"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train.py --model siamese_focal_bn --finetune --epochs 15 --lr_drop_step 5