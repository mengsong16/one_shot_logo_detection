import os
import sys
import math
import random
import shutil
import time
import numpy as np
import scipy.misc
from PIL import Image
import pandas as pd
from collections import OrderedDict

sys.path.append("../")
from config import *

# gen patch list for testset
def gen_patch_list_test(pos_patch_file, neg_patch_file, output_dir, output_file, neg_n):	

	if not os.path.exists(output_dir): 
		os.makedirs(output_dir)
		
	patch_list = []
	
	# read positive patches
	with open(pos_patch_file, 'r') as pinf:
		for line in pinf:
			items = line.strip().split(',')
			patch_list.append(str(items[1])+','+os.path.join('patches', items[0])+'\n')
			
	# sample negative patches
	with open(neg_patch_file, 'r') as ninf:
		lines = ninf.readlines()

	neg_patches = random.sample(lines, neg_n)
	for line in neg_patches:
		patch_list.append(str(0)+','+os.path.join('patches', line.strip())+'\n')

	# shuffle
	#random.shuffle(patch_list)

	# dump		
	with open(os.path.join(output_dir, output_file), 'w') as outf:
		for p in patch_list:
			outf.write(p)

	print("Done: "+output_file+", number of patches: "+str(len(patch_list)))


# gen patch list for training set
def gen_patch_list_train(pos_patch_file, neg_patch_file, clean_logo_file, output_dir, output_file, neg_n):	

	if not os.path.exists(output_dir): 
		os.makedirs(output_dir)

	patch_list = []
	
	# read positive patches
	with open(pos_patch_file, 'r') as pinf:
		for line in pinf:
			items = line.strip().split(',')
			patch_list.append(str(items[1])+','+os.path.join('patches', items[0])+'\n')
			
	# sample negative patches
	with open(neg_patch_file, 'r') as ninf:
		lines = ninf.readlines()

	neg_patches = random.sample(lines, neg_n)
	for line in neg_patches:
		patch_list.append(str(0)+','+os.path.join('patches', line.strip())+'\n')

	# read clean logos	
	with open(clean_logo_file, 'r') as cinf:
		for line in cinf:
			items = line.strip().split(',')
			patch_list.append(str(items[0])+','+os.path.join('clean_logos', items[1])+'\n')
	
	# shuffle
	random.shuffle(patch_list)

	# dump		
	with open(os.path.join(output_dir, output_file), 'w') as outf:
		for p in patch_list:
			outf.write(p)

	print("Done: "+output_file+", number of patches: "+str(len(patch_list)))

# main function 
if __name__ == '__main__':
	# redirect output 
	sys.stdout = open(os.path.join(fact_dir, 'fact_patch_for_embedding.txt'), 'w')
	neg_n = 400
	gen_patch_list_train(os.path.join(train_patch_dir, 'train_pos.txt'), os.path.join(train_patch_dir, 'train_neg.txt'), os.path.join(csv_cleanlogo_dir, 'train_cleanlogos.txt'), train_lmdb_dir, 'train_patch_list.txt', neg_n)
	gen_patch_list_test(os.path.join(test_patch_dir, 'test_w32_pos.txt'), os.path.join(test_patch_dir, 'test_w32_neg.txt'), test_lmdb_dir, 'test_w32_patch_list.txt', neg_n)
	gen_patch_list_test(os.path.join(test_patch_dir, 'test_wo32_pos.txt'), os.path.join(test_patch_dir, 'test_wo32_neg.txt'), test_lmdb_dir, 'test_wo32_patch_list.txt', neg_n)
	gen_patch_list_test(os.path.join(test_patch_dir, 'test_seen_pos.txt'), os.path.join(test_patch_dir, 'test_seen_neg.txt'), test_lmdb_dir, 'test_seen_patch_list.txt', neg_n)
	gen_patch_list_test(os.path.join(val_patch_dir, 'val_pos.txt'), os.path.join(val_patch_dir, 'val_neg.txt'), val_lmdb_dir, 'val_patch_list.txt', neg_n)
	gen_patch_list_test(os.path.join(val_patch_dir, 'val_seen_pos.txt'), os.path.join(val_patch_dir, 'val_seen_neg.txt'), val_lmdb_dir, 'val_seen_patch_list.txt', neg_n)

	

