import os
import sys
import math
import random
import shutil
import time
import numpy as np

sys.path.append("../")
from config import *

# read logo patches
# format: patch_id class_id path
def read_logo_patch_list(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()

	print('Read logo patches: '+str(len(lines)))

	return lines

def extract_groudtruth_patch(pos_list, gt_list):
	full_list = read_logo_patch_list(pos_list)
	i = 0
	with open(gt_list, 'w') as f: 
		for line in full_list:
			if "_1.jpg" in line:
				f.write(line)
				i += 1

	assert i == int(len(full_list)/5)
	print("Extract " + str(i) + " gt logo patches.")			


# main
if __name__ == '__main__':
	
	patch_dir = os.path.join(dataset_path, 'patches')

	# redirect output 
	sys.stdout = open(os.path.join(patch_dir, 'fact_gt.txt'), 'w')

	print("Extract gt patches for training set.")
	extract_groudtruth_patch(os.path.join(patch_dir, "train_pos_list.txt"), os.path.join(patch_dir, "train_pos_gt_list.txt"))
	print("Extract gt patches for test set.")
	extract_groudtruth_patch(os.path.join(patch_dir, "test_pos_list.txt"), os.path.join(patch_dir, "test_pos_gt_list.txt"))

	print("Done.")
	
