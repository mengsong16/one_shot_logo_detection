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

# clear directory
def clear_dir(folder):
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print(e)
	return

# ground truth for exact matching
def gen_pos_pair_gt_exact_matching(pos_test_pair_file, logopatch2index, cleanlogo2index, output_dir):
	with open(pos_test_pair_file, 'r') as in_file:
		lines = in_file.readlines()

	lines = [x.strip().split() for x in lines]

	n = len(lines)
	with open(os.path.join(output_dir, "pos_pair_gt.txt"), 'w') as out_file:
		i = 0
		for line in lines:
			logopatch_index = logopatch2index[line[0]]
			cleanlogo_index = cleanlogo2index[line[1]]
			out_file.write(str(logopatch_index) + " " + str(cleanlogo_index) + "\n")
			i += 1

	assert n == i
	print("Generate " + str(n) + " positive pair ground truth.")

# ground truth for same class matching
def gen_pos_pair_gt(pos_test_pair_file, logopatch2index, cleanlogo2index, output_dir):
	with open(pos_test_pair_file, 'r') as in_file:
		lines = in_file.readlines()

	lines = [x.strip().split() for x in lines]

	n = len(lines)
	with open(os.path.join(output_dir, "pos_pair_same_class_gt.txt"), 'w') as out_file:
		i = 0
		for line in lines:
			logopatch_index = logopatch2index[line[0]]
			cleanlogo_index = cleanlogo2index[line[1]]
			out_file.write(str(logopatch_index) + " " + str(cleanlogo_index) + "\n")
			i += 1

	assert n == i
	print("Generate " + str(n) + " positive pair ground truth.")

def gen_logo_patch_list(output_dir, pos_test_pair_file):
	with open(pos_test_pair_file, 'r') as in_file:
		lines = in_file.readlines()

	lines = [x.strip().split() for x in lines]

	logopatch2index = {}
	n = len(lines)
	with open(os.path.join(output_dir, "logo_patch_list.txt"), 'w') as out_file:
		i = 0
		for line in lines:
			logopatch2index[str(line[0])] = i
			out_file.write(str(line[0]) + "\n")
			i += 1

	assert n == i
	print("Get " + str(n) + " logo patches.")

	return logopatch2index

def gen_logo_patch_list_ids(output_dir, pos_test_pair_file):
	with open(pos_test_pair_file, 'r') as in_file:
		lines = in_file.readlines()

	lines = [x.strip().split() for x in lines]

	logopatch2classids = {}
	n = len(lines)
	with open(os.path.join(output_dir, "logo_patch_list.txt"), 'w') as out_file:
		i = 0
		for line in lines:
			logopatch2classids[i] = int(line[3])
			out_file.write(str(line[0]) + "\n")
			i += 1

	assert n == i
	print("Get " + str(n) + " logo patches.")
	print("Get " + str(len(logopatch2classids)) + " logo class ids.")

	return logopatch2classids

def gen_no_logo_patch_list(output_dir, neg_test_pair_file):
	with open(neg_test_pair_file, 'r') as in_file:
		lines = in_file.readlines()

	lines = [x.strip().split() for x in lines]
	
	with open(os.path.join(output_dir, "no_logo_patch_list.txt"), 'w') as out_file:
		i = 0
		for line in lines:
			if "no_logo/" in line[0]:
				out_file.write(str(line[0]) + "\n")
				i += 1

	print("Get " + str(i) + " no-logo patches.")	


def read_test_clean_logo_names(test_clean_logo_file):	
	with open(test_clean_logo_file, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)

	names = lines[:,1].tolist()
	class_ids = lines[:,0].tolist()

	print("Read "+str(len(names))+" clean logo names.")
	print("Read "+str(len(class_ids))+" clean logo ids.")
	return names, class_ids

def collect_test_clean_logo_paths(clean_logo_names, clean_logo_rel_dir, dataset_dir, output_dir):
	clean_logo_abs_dir = os.path.join(dataset_dir, "clean_logos")
	cleanlogo2index = {}

	i = 0
	with open(os.path.join(output_dir, "clean_logo_list.txt"), 'w') as out_file:
		for clean_logo_name in clean_logo_names:
			folder_abs_path = os.path.join(clean_logo_abs_dir, clean_logo_name)
			folder_rel_path = os.path.join(clean_logo_rel_dir, clean_logo_name)
			if os.path.isdir(folder_abs_path):
				for file in os.listdir(folder_abs_path):
					if file.endswith('.jpg'):
						out_file.write(os.path.join(folder_rel_path, file)+"\n")
						cleanlogo2index[os.path.join(folder_rel_path, file)] = i
						i += 1		
			else:
				print("Error: "+folder_abs_path+" Does NOT exist.")
				break

	print("Get "+str(i)+" clean logos.")

	return cleanlogo2index	

def dump_class_id_for_pos_patches_clean_logos(cleanlogo2classids, logopatch2classids, output_dir):
	with open(os.path.join(output_dir, "clean_logo_class_ids.txt"), 'w') as out_file_clean_logo:
		for i, cleanlogoid in cleanlogo2classids.items():
			out_file_clean_logo.write(str(cleanlogoid)+"\n")

	with open(os.path.join(output_dir, "logo_patch_class_ids.txt"), 'w') as out_file_logo_patch:
		for i, logpatchid in logopatch2classids.items():
			out_file_logo_patch.write(str(logpatchid)+"\n")		

def collect_test_clean_logo_paths_ids(clean_logo_names, clean_logo_class_ids, clean_logo_rel_dir, dataset_dir, output_dir):
	clean_logo_abs_dir = os.path.join(dataset_dir, "clean_logos")
	cleanlogo2classids = {}

	i = 0
	with open(os.path.join(output_dir, "clean_logo_list.txt"), 'w') as out_file:
		c = 0
		for clean_logo_name in clean_logo_names:
			logo_class_id = clean_logo_class_ids[c]
			folder_abs_path = os.path.join(clean_logo_abs_dir, clean_logo_name)
			folder_rel_path = os.path.join(clean_logo_rel_dir, clean_logo_name)
			if os.path.isdir(folder_abs_path):
				for file in os.listdir(folder_abs_path):
					if file.endswith('.jpg'):
						out_file.write(os.path.join(folder_rel_path, file)+"\n")
						cleanlogo2classids[i] = int(logo_class_id)
						i += 1		
			else:
				print("Error: "+folder_abs_path+" Does NOT exist.")
				break

			c += 1	

	assert i == len(cleanlogo2classids)		
	print("Get "+str(i)+" clean logos and their class ids.")

	return cleanlogo2classids	

# main
if __name__ == '__main__':

	#dataset_dir = '/work/meng/data/logosc_300'
	# which test files do we want to use
	pair_dir = 'pairs_gt'
	
	neg_test_pair_file = os.path.join(dataset_path, pair_dir+'/test_neg.txt')
	test_clean_logo_file = os.path.join(dataset_path, 'split/test_class.txt')
	clean_logo_rel_dir = "clean_logos"

	output_dir = os.path.join(dataset_path, 'embedding_test')

	option = "same_class"

	
	# creat or clear output dir
	if not os.path.exists(output_dir): 
		os.makedirs(output_dir)
	#else:	
	#	clear_dir(output_dir)

	# redirect output 
	sys.stdout = open(os.path.join(output_dir, 'fact.txt'), 'w')

	print("Generating from "+pair_dir)
	
	# get no-logo patch list
	gen_no_logo_patch_list(output_dir, neg_test_pair_file)

	if option == "exact_match":
		# get clean logo names
		clean_logo_names, _ = read_test_clean_logo_names(test_clean_logo_file)
		cleanlogo2index = collect_test_clean_logo_paths(clean_logo_names, clean_logo_rel_dir, dataset_path, output_dir)
		# get logo patch list	
		pos_test_pair_file = os.path.join(dataset_path, pair_dir+'/test_pos.txt')
		logopatch2index = gen_logo_patch_list(output_dir, pos_test_pair_file)
		# get ground truth of class matching for positive pairs
		gen_pos_pair_gt(pos_test_pair_file, logopatch2index, cleanlogo2index, output_dir)
	else:
		# get clean logo names and class ids
		clean_logo_names, clean_logo_class_ids = read_test_clean_logo_names(test_clean_logo_file)
		cleanlogo2classids = collect_test_clean_logo_paths_ids(clean_logo_names, clean_logo_class_ids, clean_logo_rel_dir, dataset_path, output_dir)	
		# get logo patch list	
		pos_test_pair_file = os.path.join(dataset_path, pair_dir+'/test_pos_hm.txt')
		logopatch2classids = gen_logo_patch_list_ids(output_dir, pos_test_pair_file)

		dump_class_id_for_pos_patches_clean_logos(cleanlogo2classids, logopatch2classids, output_dir)


	