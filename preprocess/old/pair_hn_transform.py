import os
import sys
import math
import random
import shutil
import time
import numpy as np
import datetime
import re

sys.path.append("../")
from config import *

def split_neg_from_full(pair_dir, spec):
	full_file_name = os.path.join(pair_dir, spec+".txt")
	with open(full_file_name, 'r') as ff:
		full_lines = ff.readlines()

	pos_file_name = os.path.join(pair_dir, spec+"_pos.txt")
	with open(pos_file_name, 'r') as pf:
		pos_lines = pf.readlines()

	neg_list = list(set(full_lines)-set(pos_lines))
	neg_list = [x.strip().split() for x in neg_list]
	neg_file_name = os.path.join(pair_dir, spec+"_neg.txt")
	with open(neg_file_name, 'w') as nf:
		for line in neg_list:
			nf.write(line[0] + ' ' + line[1] + ' ' + line[2] + '\n')

	print('Full:' + str(len(full_lines)))
	print('Pos:' + str(len(pos_lines)))
	print('Neg:' + str(len(neg_list)))


# include class 0 no_logo
def read_class_map(class_list):	
	class_map = np.load(class_list).item()
	class_map[0] = "no_logo"

	#print(class_map)
	return class_map

def reverse_class_map(class_map):
	class_map_reversed = {}
	for k, v in class_map.items():
		class_map_reversed[v] = k

	return class_map_reversed	

def path2classid(line, class_map_reversed):
	pos = list(re.finditer('/', line))
	class_name = line[pos[0].start()+1:pos[1].start()]
	return class_map_reversed[class_name]
	

def add_class_id(pair_dir, src_filename, class_map_reversed):

	src_file = os.path.join(pair_dir, src_filename+".txt")
	des_file = os.path.join(pair_dir, src_filename+"_hm.txt")

	with open(src_file, 'r') as sf:
		lines = sf.readlines()

	lines = [x.strip().split() for x in lines]
	with open(des_file, 'w') as df:	
		for line in lines:
			source_id = path2classid(line[0], class_map_reversed)
			# positive pair
			if int(line[2]) == 1:
				target_id = source_id
			# negative pair	
			else:	
				target_id = path2classid(line[1], class_map_reversed)

			# write a new line with class ids
			df.write(line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + str(source_id) + ' ' + str(target_id) + '\n')	

# main
if __name__ == '__main__':

	#root_dir = '/work/meng/data/logosc_300'
	#pair_dir = os.path.join(root_dir, 'pairs')	
	#pair_dir = os.path.join(root_dir, 'pairs_val')	
	pair_dir = os.path.join(dataset_path, 'pairs_gt')	
	class_list = os.path.join(dataset_path, 'split/class_map.npy')

	class_map = read_class_map(class_list)
	#print(class_map)
	class_map_reversed = reverse_class_map(class_map)

	# add_class_id(pair_dir, "train_neg", class_map_reversed)
	# add_class_id(pair_dir, "train_pos", class_map_reversed)
	# add_class_id(pair_dir, "test_neg", class_map_reversed)
	# add_class_id(pair_dir, "test_pos", class_map_reversed)
	
	#path2classid("patches/Bass_Pro_Shops/earthday17~7db0bcf969611367828fd15a84379d92_1.jpg", class_map_reversed)
	#split_neg_from_full(os.path.join(root_dir, pair_path), 'test')
	#split_neg_from_full(os.path.join(root_dir, pair_path), 'train')

	'''
	split_neg_from_full(os.path.join(root_dir, pair_path), 'val')
	split_neg_from_full(os.path.join(root_dir, pair_path), 'train')
	add_class_id(pair_dir, "train_neg", class_map_reversed)
	add_class_id(pair_dir, "train_pos", class_map_reversed)
	add_class_id(pair_dir, "val_neg", class_map_reversed)
	add_class_id(pair_dir, "val_pos", class_map_reversed)
	'''

	split_neg_from_full(pair_dir, 'test')
	split_neg_from_full(pair_dir, 'train')
	add_class_id(pair_dir, "train", class_map_reversed)
	add_class_id(pair_dir, "train_neg", class_map_reversed)
	add_class_id(pair_dir, "train_pos", class_map_reversed)
	add_class_id(pair_dir, "test", class_map_reversed)
	add_class_id(pair_dir, "test_neg", class_map_reversed)
	add_class_id(pair_dir, "test_pos", class_map_reversed)