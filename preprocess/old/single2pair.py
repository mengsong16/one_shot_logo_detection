import os
import sys
import math
import random
import shutil
import time
import numpy as np

sys.path.append("../")
from config import *

# read mapping between the image and the right clean logo 
def read_clean_logo_map(clean_logo_list):
	clean_logo_map = np.load(clean_logo_list).item()

	return clean_logo_map


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

# read logo patches
# format: patch_id class_id path
def read_logo_patches(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	print('Read logo patches: '+str(np.shape(lines)[0]))
	return lines[:,1], lines[:,2]

# for each class, gather logo patches
def build_logo_bins(patch_dir, spec):
	#file_name = os.path.join(patch_dir, spec+'_pos_list.txt')
	file_name = os.path.join(patch_dir, spec+'_pos_gt_list.txt')
	print("Constructing logo bins for "+file_name)
	with open(file_name, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	lines = lines[:, 1:3]
	N = np.shape(lines)[0]
	print('Read logo patches: '+str(N))

	print("class_id: # pos patches")
	logo_bins = {}
	for line in lines:
		class_id = int(line[0])
		path = line[1]
		if class_id not in logo_bins:
			logo_bins[class_id] = [path]
		else:	
			logo_bins[class_id].append(path)

	n = 0
	c = 0
	for k, v in logo_bins.items():
		n += len(v)
		c += 1
		print(str(k)+": "+str(len(v)))

	print("Classes: "+str(c))
	assert n == N
	print("Patches: "+str(n))

	return logo_bins

def patch2cleanlogo(patch_path, clean_logo_map):
	# extract image name from patch_path
	last_occ = patch_path.rfind('_')
	image_path = patch_path[:last_occ]
	clean_logo_index = clean_logo_map[image_path]

	return clean_logo_index


# read no logo patches
# format: patch_id path
def read_no_logo_patches(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	print('Read no logo patches: '+str(np.shape(lines)[0]))
	return lines[:,1]

# include test classes and training classes
def read_class_map(class_list):
	class_map = np.load(class_list).item()
	# map no_logo to 0
	# map other classes to positive integers 
	class_map[0] = 'no_logo'
	return class_map


# dump a list of strings to a file
def dump(l, file):
	print("Dump to " + file)
	with open(file, 'w') as f:
		for line in l:
			f.write(line)
		
	print("Done.")

# generate positive pairs
# format: source target label
def generate_pos_pairs(class_list, root_dir, patch_dir, clean_logo_path, clean_logo_map, spec):
	pos_pairs = []
	class_map = read_class_map(class_list)
	#class_ids, image_paths = read_logo_patches(os.path.join(os.path.join(root_dir, patch_dir), spec+'_pos_list.txt'))
	class_ids, image_paths = read_logo_patches(os.path.join(os.path.join(root_dir, patch_dir), spec+'_pos_gt_list.txt'))
	N = np.shape(image_paths)[0]

	for i in list(range(N)):
		class_id = int(class_ids[i])
		class_name = class_map[class_id]
		clean_logo_folder = os.path.join(clean_logo_path, class_name)
		pos_patch_full_path = os.path.join(patch_dir, image_paths[i])
		clean_logo_full_path = os.path.join(clean_logo_folder, patch2cleanlogo(image_paths[i], clean_logo_map))

		# check correctness
		assert(os.path.exists(os.path.join(root_dir, pos_patch_full_path)))
		assert(os.path.exists(os.path.join(root_dir, clean_logo_full_path)))

		pos_pairs.append(pos_patch_full_path + ' ' + clean_logo_full_path + ' ' + str(1) + '\n')

	assert len(pos_pairs) == (i+1)
	print("Generated "+str(len(pos_pairs))+" positive pairs.")
	
	return pos_pairs

# for each class, compute how many logo patch and no-logo patch we need
def split_logo_bins(logo_bins, logo_rate):
	no_logo_counts = {}
	logo_counts = {}
	if logo_rate == 1:
		for k, v in logo_bins.items():
			logo_counts[k] = len(v)

	elif logo_rate == 0:
		for k, v in logo_bins.items():
			no_logo_counts[k] = len(v)
	else:
		for k, v in logo_bins.items():
			logo_n = int(math.ceil(len(v)*logo_rate))
			logo_counts[k] = logo_n
			no_logo_counts[k] = len(v) - logo_n

	return logo_counts, no_logo_counts		

# get all logo patches except one given class
def get_rest_classes(class_id, logo_bins):
	rest_list = []
	s = 0
	for cls_id, paths in logo_bins.items():
		s += len(paths)
		if cls_id != class_id:
			rest_list += paths

	assert s == len(rest_list) + len(logo_bins[class_id])

	return rest_list

def get_logo_n(clean_logo_folder_path):
	i = 0
	for logo in os.listdir(clean_logo_folder_path):
		if logo.endswith(".jpg"):
			i += 1

	return i		

def generate_neg_logo_pairs(class_list, patch_dir, root_dir, clean_logo_path, logo_bins, logo_counts, clean_logo_map, spec):
	neg_logo_pairs = []
	class_map = read_class_map(class_list)
	i = 0
	for class_id, paths in logo_bins.items():
		# uniformly sample among the rest classes
		rest_list = get_rest_classes(class_id, logo_bins)
		sample_n = logo_counts[class_id]
		assert len(rest_list) >= sample_n
		sampled_list = random.sample(rest_list, sample_n)

		class_name = class_map[class_id]
		clean_logo_folder = os.path.join(clean_logo_path, class_name)
		N = get_logo_n(os.path.join(root_dir, clean_logo_folder))
		
		for sample in sampled_list:
			neg_patch_full_path = os.path.join(patch_dir, sample)
			clean_logo_full_path = os.path.join(clean_logo_folder, str(random.randint(0,N-1))+'.jpg')

			# check correctness
			assert(os.path.exists(os.path.join(root_dir, neg_patch_full_path)))
			assert(os.path.exists(os.path.join(root_dir, clean_logo_full_path)))


			line = neg_patch_full_path + ' ' + clean_logo_full_path + ' ' + str(0) + '\n'
			neg_logo_pairs.append(line)
			i += 1

	
	assert len(neg_logo_pairs) == i
	print("Generated "+str(len(neg_logo_pairs))+" logo negative pairs.")
	return neg_logo_pairs

def generate_neg_no_logo_pairs(class_list, root_dir, patch_dir, clean_logo_path, no_logo_counts, clean_logo_map, spec):
	neg_no_logo_pairs = []
	no_logo_list = read_no_logo_patches(os.path.join(os.path.join(root_dir, patch_dir), spec+'_neg_list.txt'))
	no_logo_list = no_logo_list.tolist()
	class_map = read_class_map(class_list)
	# get sum
	s = 0
	for class_id, n in no_logo_counts.items():
		s += n
	# uniformly sample s no-logo patches
	assert len(no_logo_list) >= s
	sampled_list = random.sample(no_logo_list, s)
	
	# assign them to each class
	i = 0	
	for class_id, n in no_logo_counts.items():
		class_name = class_map[class_id]
		clean_logo_folder = os.path.join(clean_logo_path, class_name)
		N = get_logo_n(os.path.join(root_dir, clean_logo_folder))
		
		for j in list(range(n)):
			neg_patch_full_path = os.path.join(patch_dir, sampled_list[i])
			clean_logo_full_path = os.path.join(clean_logo_folder, str(random.randint(0,N-1))+'.jpg')

			# check correctness
			assert(os.path.exists(os.path.join(root_dir, neg_patch_full_path)))
			assert(os.path.exists(os.path.join(root_dir, clean_logo_full_path)))

			line = neg_patch_full_path + ' ' + clean_logo_full_path + ' ' + str(0) + '\n'
			neg_no_logo_pairs.append(line)
			i += 1

	assert len(neg_no_logo_pairs) == i
	print("Generated "+str(len(neg_no_logo_pairs))+" no-logo negative pairs.")
	
	return neg_no_logo_pairs


# generate negative pairs
# format: source target label
def generate_neg_pairs(class_list, root_dir, patch_dir, clean_logo_path, logo_rate, clean_logo_map, spec):
	logo_bins = build_logo_bins(os.path.join(root_dir, patch_dir), spec)
	logo_counts, no_logo_counts = split_logo_bins(logo_bins, logo_rate)
	if len(logo_counts) > 0:
		neg_logo_pairs = generate_neg_logo_pairs(class_list, patch_dir, root_dir, clean_logo_path, logo_bins, logo_counts, clean_logo_map, spec)
	else:
		neg_logo_pairs = []
	if len(no_logo_counts) > 0:	
		neg_no_logo_pairs = generate_neg_no_logo_pairs(class_list, root_dir, patch_dir, clean_logo_path, no_logo_counts, clean_logo_map, spec)
	else:
		neg_no_logo_pairs = []
	
	return neg_logo_pairs + neg_no_logo_pairs

# main
if __name__ == '__main__':
	### Should change related path for option gt #####
	#root_dir = '/work/meng/data/logosc_300'
	class_list = os.path.join(dataset_path, 'split/class_map.npy')
	patch_dir = 'patches'
	#pair_dir = os.path.join(root_dir, 'pairs')
	pair_dir = os.path.join(dataset_path, 'pairs_gt')
	clean_logo_path = 'clean_logos'
	clean_logo_png_path = 'clean_logos_png'

	logo_rate = 0.5

	if not os.path.exists(pair_dir): 
		os.makedirs(pair_dir)

	# redirect output 
	sys.stdout = open(os.path.join(pair_dir, 'fact.txt'), 'w')

	# get clean logo map 
	clean_logo_map = read_clean_logo_map(os.path.join(os.path.join(dataset_path, clean_logo_png_path), 'clean_logo_map.npy'))

	# generate
	train_pos_list = generate_pos_pairs(class_list, dataset_path, patch_dir, clean_logo_path, clean_logo_map, 'train')
	test_pos_list = generate_pos_pairs(class_list, dataset_path, patch_dir, clean_logo_path, clean_logo_map, 'test')
	train_neg_list = generate_neg_pairs(class_list, dataset_path, patch_dir, clean_logo_path, logo_rate, clean_logo_map, 'train')
	test_neg_list = generate_neg_pairs(class_list, dataset_path, patch_dir, clean_logo_path, logo_rate, clean_logo_map, 'test')
	# save 
	dump(train_pos_list, os.path.join(pair_dir, 'train_pos.txt'))
	dump(test_pos_list, os.path.join(pair_dir, 'test_pos.txt'))
	dump(train_pos_list+train_neg_list, os.path.join(pair_dir, 'train.txt'))
	dump(test_pos_list+test_neg_list, os.path.join(pair_dir, 'test.txt'))
	
