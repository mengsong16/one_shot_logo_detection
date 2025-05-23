import os
import sys
import math
import random
import shutil
import time
import numpy as np
import sys

sys.path.append("../")
from config import *

# lines are narray
def build_pair_bins(lines):
	N = np.shape(lines)[0]
	
	bins = {}
	for line in lines:
		target_class_id = int(line[4])
		# classify by target_class_id
		if target_class_id not in bins:
			bins[target_class_id] = [line]
		else:	
			bins[target_class_id].append(line)

	# check correctness		
	n = 0
	for k, v in bins.items():
		n += len(v)
	
	assert n == N
	print('Read and classify pairs: '+str(n))
	print('Class num: '+str(len(bins)))

	return bins

def split_pos_pairs(pair_file, train_rate):
	# construct class bins, class_id is indexed from 1
	with open(pair_file, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)

	pos_bins = build_pair_bins(lines)

	train = []
	val = []
	# v is array
	for k, v in pos_bins.items():
		train_n = int(round(len(v) * train_rate))
		random.shuffle(v)
		train += v[:train_n]
		val += v[train_n:]

	assert len(train)+len(val) == len(lines)

	print("Split: train="+str(len(train))+", val="+str(len(val)))
	
	return train, val	

def split_neg_pairs(pair_file, train_rate):
	with open(pair_file, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)	
	# split into no-logo and logo
	no_logo = []
	logo = []
	for line in lines:
		source_class_id = int(line[3])	
		if source_class_id == 0:
			no_logo.append(line)
		else:
			logo.append(line)

	assert len(logo)+len(no_logo) == len(lines)
	print("Split: logo="+str(len(logo))+", no_logo="+str(len(no_logo)))
	
	train = []
	val = []
	# split no-logo pairs
	train_n = int(round(len(no_logo) * train_rate))	
	random.shuffle(no_logo)
	train += no_logo[:train_n]
	val += no_logo[train_n:]
	# split logo pairs
	neg_bins = build_pair_bins(logo)
	for k, v in neg_bins.items():
		train_n = int(round(len(v) * train_rate))
		random.shuffle(v)
		train += v[:train_n]
		val += v[train_n:]

	assert len(train)+len(val) == len(lines)

	print("Split: train="+str(len(train))+", val="+str(len(val)))
	return train, val

def remove_class_id(lines):
	lines = np.asarray(lines)
	return lines[:,:3]

def read_train_file(train_file, train_rate):
	with open(train_file, 'r') as f:
		lines = f.readlines()

	total_n = len(lines)

	print("Read "+str(total_n)+" pairs.")

	train_n = int(math.floor(total_n * train_rate))
	
	random.shuffle(lines)
	train = lines[:train_n]
	val = lines[train_n:]

	print("Read file: "+train_file)
	print("Split into train = "+str(len(train))+", val = "+str(len(val)))
	
	return train, val

def dump(lines, file):
	with open(file, 'w') as f:
		n = np.shape(lines)[1]
		for l in lines:
			s = ""
			for i in list(range(n-1)):
				s += (str(l[i]) + " ")

			s += (str(l[n-1]) + "\n")	

			f.write(s)
				
	print("Dumped file: "+file)	
		
# main
if __name__ == '__main__':
	#root_dir = '/work/meng/data/logosc_300'	
	#image_dir = os.path.join(root_dir, 'logosc300')
	output_dir = os.path.join(dataset_path, 'pairs_val')
	train_rate = 0.9

	if not os.path.exists(output_dir): 
		os.makedirs(output_dir)
	
	sys.stdout = open(os.path.join(output_dir, 'fact.txt'), 'w')

	print("split positive pairs...")
	pos_train, pos_val = split_pos_pairs(os.path.join(output_dir, "train_all_pos_hm.txt"), train_rate)
	print("split negative pairs...")
	neg_train, neg_val = split_neg_pairs(os.path.join(output_dir, "train_all_neg_hm.txt"), train_rate)

	train = pos_train + neg_train
	val = pos_val + neg_val

	print("==> train: "+str(len(train)))
	print("==> val: "+str(len(val)))
	
	# get pairs without class ids
	train_short = remove_class_id(train)
	val_short = remove_class_id(val)
	pos_train_short = remove_class_id(pos_train)
	pos_val_short = remove_class_id(pos_val)
	neg_train_short = remove_class_id(neg_train)
	neg_val_short = remove_class_id(neg_val)

	# dump
	print("Dumping...")
	dump(train_short, os.path.join(output_dir, "train.txt"))
	dump(pos_train_short, os.path.join(output_dir, "train_pos.txt"))
	dump(neg_train_short, os.path.join(output_dir, "train_neg.txt"))
	dump(val_short, os.path.join(output_dir, "val.txt"))
	dump(pos_val_short, os.path.join(output_dir, "val_pos.txt"))
	dump(neg_val_short, os.path.join(output_dir, "val_neg.txt"))

	dump(train, os.path.join(output_dir, "train_hm.txt"))
	dump(pos_train, os.path.join(output_dir, "train_pos_hm.txt"))
	dump(neg_train, os.path.join(output_dir, "train_neg_hm.txt"))
	dump(val, os.path.join(output_dir, "val_hm.txt"))
	dump(pos_val, os.path.join(output_dir, "val_pos_hm.txt"))
	dump(neg_val, os.path.join(output_dir, "val_neg_hm.txt"))

	print("Done.")

	

	

	