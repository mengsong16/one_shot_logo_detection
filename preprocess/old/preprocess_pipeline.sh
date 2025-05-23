#! /bin/bash


## Run this script using command "bash ~.sh"

## preprocess
cd /work/meng/uvn/preprocess
python check_dataset.py
python split.py
python get_min_bbox.py
#python copy_clean_logos.py
python convert_clean_logos.py
python generate_clean_logos.py
python compute_image_mean.py
python generate_patches.py
#python find_right_clean_logo.py
python single2pair.py
