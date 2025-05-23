import os
import sys
import math
import random
import shutil
import time
import numpy as np
import re

# for each class name, gather clean logo paths
def build_clean_logo_bins(dataset_dir, clean_logo_dir, train_class_file):
	# read training classes
	with open(train_class_file, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)

	n = lines.shape[0]
	train_class_names = []
	for i in list(range(n)):
		train_class_names.append(lines[i][1])

	clean_logo_bins = {}

	clean_logo_abs_dir = os.path.join(dataset_dir, clean_logo_dir)
	for folder in train_class_names:
		folder_full = os.path.join(clean_logo_dir, folder)
		folder_full_abs = os.path.join(clean_logo_abs_dir, folder)
		if os.path.isdir(folder_full_abs):
			# for all images in this brand, collect paths
			for file in os.listdir(folder_full_abs):
				if file.endswith('.jpg'):
					image_path = os.path.join(folder_full, file)
					if folder not in clean_logo_bins:
						clean_logo_bins[folder] = [image_path]
					else:
						clean_logo_bins[folder].append(image_path)

	n = 0
	c = 0
	for k, v in clean_logo_bins.items():
		#print(k+": "+v[0])
		n += len(v)
		c += 1

	print("Clean logo bins:")
	print("Classes: "+str(c))
	print("Patches: "+str(n))

	return clean_logo_bins

# get all clean logos except one given class
def get_rest_clean_logos(class_name, clean_logo_bins):
	rest_list = []
	s = 0
	for cls_name, paths in clean_logo_bins.items():
		s += len(paths)
		if cls_name != class_name:
			rest_list += paths

	assert s == len(rest_list) + len(clean_logo_bins[class_name])

	return rest_list

def get_possible_negs(pos_path, rest_clean_logo_paths):
	negs = []
	for cln_path in rest_clean_logo_paths:
		#negs += (pos_path + " " + cln_path + " " + str(0) + "\n")
		negs.append((str(pos_path), str(cln_path), int(0)))

	return negs	

def get_class_name(path):
	pos = list(re.finditer('/', path))
	class_name = path[pos[0].start()+1:pos[1].start()]
	return class_name

def mine_for_all(pos_pair_file, clean_logo_bins):
	with open(pos_pair_file, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)

	
	for line in lines:
		class_name = get_class_name(line[0])
		rest_clean_logo_paths = get_rest_clean_logos(class_name, clean_logo_bins)
		negs = get_possible_negs(line[0], rest_clean_logo_paths)


	
		
# main function 
if __name__ == '__main__':

	dataset_dir = '/work/meng/data/logosc_300'
	clean_logo_dir = 'clean_logos'
	pos_pair_file = os.path.join(os.path.join(dataset_dir, "pairs"), "train_pos.txt")
	train_class_file = os.path.join(os.path.join(dataset_dir, "split"), "train_class.txt")

	clean_logo_bins = build_clean_logo_bins(dataset_dir, clean_logo_dir, train_class_file)
	#print(len(get_rest_clean_logos("New_Belgium", clean_logo_bins)))
	mine_for_all(pos_pair_file, clean_logo_bins)