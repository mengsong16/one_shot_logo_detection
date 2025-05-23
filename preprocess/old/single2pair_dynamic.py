import os
import sys
import math
import random
import shutil
import time
import numpy as np
import datetime

sys.path.append("../")
from config import *

def class_map_txt_to_npy(in_txt, out_npy):
	with open(in_txt, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	class_map = {}
	for line in lines:
		class_map[int(line[0])] = line[1]

	print(str(len(class_map)))
	
	np.save(out_npy, class_map)	

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
def build_logo_bins(file_name):
	#file_name = os.path.join(patch_dir, spec+'_pos_list.txt')
	print("Constructing logo bins from logo patch list: "+file_name)

	with open(file_name, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	lines = lines[:, 1:3]

	N = np.shape(lines)[0]
	#print('Read logo patches: '+str(N))

	#print("class_id: # pos patches")
	logo_bins = {}
	for line in lines:
		class_id = int(line[0])
		path = line[1]
		if class_id not in logo_bins:
			logo_bins[class_id] = [path]
		else:	
			logo_bins[class_id].append(path)

	n = 0
	for k, v in logo_bins.items():
		n += len(v)
		#print(str(k)+": "+str(len(v)))

	#print("Classes: "+str(len(logo_bins)))
	assert n == N
	#print("Patches: "+str(n))

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
	return lines[:,1].tolist()


# not include test classes and training classes
def read_class_map_pos(class_list):
	class_map = np.load(class_list).item()
	#print("Loaded "+str(len(class_map))+" classes.")
	return class_map

# dump a list of strings to a file
def dump(l, file):
	print("Dump to " + file)
	with open(file, 'w') as f:
		for line in l:
			f.write(line)
		
	print("Done.")

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

def sample_rest_classes_without_leaving_out(class_id, root_dir, patch_dir, spec, logo_bins, logo_counts, sample_n, fixed_logo_bins):
	#print("==> Sampling negative pairs for class "+str(class_id))
	# get the number of all rest samples
	rest_n = 0
	for k, count in logo_counts.items():
		if k != class_id:
			rest_n += count

	# count number of samples from each class
	sample_count = {}
	rest_class_ids = []
	n = 0
	for cls_id, paths in logo_bins.items():
		if cls_id != class_id:
			# how many samples from this class
			# logo_counts will keep the same forever
			rest_class_ids.append(cls_id)
			sample_count[cls_id] = int(math.floor(float(sample_n)*(float(logo_counts[cls_id])/float(rest_n))))

			if sample_count[cls_id] < 1:
				sample_count[cls_id] = 1

			n += sample_count[cls_id]	

	r = sample_n - n

	#print(str(r)+" "+str(sample_n))

	# need to sample more
	if r > 0:
		assert r <= len(rest_class_ids)
		r_classes = random.sample(rest_class_ids, r)
		for c in r_classes:
			sample_count[c] += 1
	# need to remove some samples		
	elif r < 0:
		assert -r <= len(rest_class_ids)
		cids = random.sample(rest_class_ids, -r)
		for cid in cids:
			if sample_count[cid] > 1:
				sample_count[cid] -= 1
			else:
				_ = sample_count.pop(cid)
		
	# check correctness			
	cn = 0;
	for k, v in sample_count.items():
		cn += v

	assert cn == sample_n				

	# sample from all rest classes
	sampled_list = []
			
			
	for cls_id, num in sample_count.items():
		paths = logo_bins[cls_id]
		
		#print("Sampling class "+str(cls_id))
		# reload
		if num > len(paths):
			paths = reload_one_class(cls_id, fixed_logo_bins)
			logo_bins[cls_id] = paths
			#print("Reload class "+str(cls_id))
		# sample from current class
		full_path_list = list(range(len(paths)))
		sampled_path_index_list = random.sample(full_path_list, num)
		sampled_list += [paths[ind] for ind in sampled_path_index_list]
		#print(sampled_list)
		full_path_set = set(full_path_list)
		sampled_path_index_set = set(sampled_path_index_list)
		rest_list = list(full_path_set - sampled_path_index_set)
		# remove sampled ones from logo_bins
		#print("Before removal: "+str(len(logo_bins[cls_id])))
		#print("Sampled: "+str(sample_count[cls_id]))
		logo_bins[cls_id] = [paths[ind] for ind in rest_list]
		#print("After removal: "+str(len(logo_bins[cls_id])))	

	assert 	len(sampled_list) == sample_n	
	#print("==> Sampled negative pairs: "+str(len(sampled_list)))
		

	return sampled_list, logo_bins


def get_logo_n(clean_logo_folder_path):
	i = 0
	for logo in os.listdir(clean_logo_folder_path):
		if logo.endswith(".jpg"):
			i += 1

	return i		

def reload_one_class(class_id, fixed_logo_bins):
	return fixed_logo_bins[class_id]

def generate_neg_logo_pairs(patch_dir, root_dir, clean_logo_path, logo_bins, logo_counts, clean_logo_map, class_map, fixed_logo_bins, spec):
	start_time = time.time()
	neg_logo_pairs = []
	i = 0
	for class_id, sample_n in logo_counts.items():
		#print(str(class_id))
		class_name = class_map[class_id]
		# uniformly sample among the rest classes
		sampled_list, logo_bins = sample_rest_classes_without_leaving_out(class_id, root_dir, patch_dir, spec, logo_bins, logo_counts, sample_n, fixed_logo_bins)

		clean_logo_folder = os.path.join(clean_logo_path, class_name)
		# how many clean logos this brand has
		N = get_logo_n(os.path.join(root_dir, clean_logo_folder))
		
		for sample in sampled_list:
			neg_patch_full_path = os.path.join(patch_dir, sample)
			clean_logo_full_path = os.path.join(clean_logo_folder, str(random.randint(0,N-1))+'.jpg')

			# check correctness
			assert(os.path.exists(os.path.join(root_dir, neg_patch_full_path)))
			assert(os.path.exists(os.path.join(root_dir, clean_logo_full_path)))

			#line = neg_patch_full_path + ' ' + clean_logo_full_path + ' ' + str(0) + '\n'
			neg_logo_pairs.append((neg_patch_full_path, clean_logo_full_path, 0))
			i += 1
	
	
	assert len(neg_logo_pairs) == i
	print("==> Generated "+str(len(neg_logo_pairs))+" logo negative pairs in total.")
	print('==> Time: '+str(datetime.timedelta(seconds=time.time()-start_time)))
	return neg_logo_pairs, logo_bins

def generate_neg_no_logo_pairs(class_list, root_dir, patch_dir, clean_logo_path, no_logo_counts, clean_logo_map, no_logo_list, class_map, spec):
	start_time = time.time()
	print("==> Sampling no-logo negative pairs...")
	neg_no_logo_pairs = []
	
	# get sum
	s = 0
	for class_id, n in no_logo_counts.items():
		s += n
	# reload no logo list
	if len(no_logo_list) < s:
		no_logo_list = read_no_logo_patches(os.path.join(os.path.join(root_dir, patch_dir), spec+'_neg_list.txt'))	
		#print("Reload no logo patches: "+str(len(no_logo_list)))
	# uniformly sample s no-logo patches
	selected_indices = random.sample(list(range(len(no_logo_list))), s)	
	sampled_list = [ no_logo_list[ind] for ind in selected_indices ]
	# remove sampled patches for no_logo_list
	#print("Before removal: "+str(len(no_logo_list)))
	#print("Sampled: "+str(len(sampled_list)))
	sampled_set = set(selected_indices)
	assert len(sampled_set) == len(sampled_list)
	full_indices = set(list(range(len(no_logo_list))))
	rest_indices = list(full_indices - sampled_set)
	assert len(rest_indices) + len(sampled_list) == len(no_logo_list)
	no_logo_list = [no_logo_list[ind] for ind in rest_indices]
	#print("After removal: "+str(len(no_logo_list)))	
	
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

			#line = neg_patch_full_path + ' ' + clean_logo_full_path + ' ' + str(0) + '\n'
			neg_no_logo_pairs.append((neg_patch_full_path, clean_logo_full_path, 0))
			i += 1

	assert len(neg_no_logo_pairs) == i
	print("==> Generated "+str(len(neg_no_logo_pairs))+" no-logo negative pairs.")
	print('==> Time: '+str(datetime.timedelta(seconds=time.time()-start_time)))
	
	return neg_no_logo_pairs, no_logo_list


# generate negative pairs
# format: source target label
def generate_neg_pairs(class_list, root_dir, patch_dir, clean_logo_path, logo_rate, clean_logo_map, logo_bins, no_logo_list, 
	logo_counts, no_logo_counts, class_map, fixed_logo_bins, spec):
	
	if len(logo_counts) > 0:
		neg_logo_pairs, logo_bins = generate_neg_logo_pairs(patch_dir, root_dir, clean_logo_path, logo_bins, logo_counts, 
			clean_logo_map, class_map, fixed_logo_bins, spec)
	else:
		neg_logo_pairs = []
	

	if len(no_logo_counts) > 0:	
		neg_no_logo_pairs, no_logo_list = generate_neg_no_logo_pairs(class_list, root_dir, patch_dir, clean_logo_path, no_logo_counts, 
			clean_logo_map, no_logo_list, class_map, spec)
	else:
		neg_no_logo_pairs = []
	
	
	return neg_logo_pairs + neg_no_logo_pairs, logo_bins, no_logo_list

# main
if __name__ == '__main__':
	
	#root_dir = '/work/meng/data/logosc_300'
	
	class_list = os.path.join(dataset_path, 'split/class_map.npy')
	patch_dir = 'patches'
	pair_dir = os.path.join(dataset_path, 'pairs_dyn')
	clean_logo_path = 'clean_logos'
	clean_logo_png_path = 'clean_logos_png'
	
	logo_rate = 0.5

	if not os.path.exists(pair_dir): 
		os.makedirs(pair_dir)

	# redirect output 
	#sys.stdout = open(os.path.join(pair_dir, 'fact.txt'), 'w')

	# get clean logo map 
	clean_logo_map = read_clean_logo_map(os.path.join(os.path.join(dataset_path, clean_logo_png_path), 'clean_logo_map.npy'))
	# logo bins tell us for each class, what images do we have, should be modified dynamically
	logo_bins = build_logo_bins(os.path.join(os.path.join(dataset_path, patch_dir), 'train_pos_list.txt'))
	fixed_logo_bins = build_logo_bins(os.path.join(os.path.join(dataset_path, patch_dir), 'train_pos_list.txt'))
	# logo and no logo counts tell us for each class, how many logo negative pairs and no logo negative pairs do we need, should keep unchanged 
	logo_counts, no_logo_counts = split_logo_bins(fixed_logo_bins, logo_rate)
	# list of no logo patches
	no_logo_list = read_no_logo_patches(os.path.join(os.path.join(dataset_path, patch_dir), 'train_neg_list.txt'))
	# map of class id and class name
	class_map = read_class_map_pos(class_list)
	# generate negative pairs repeatedly while training
	for i in range(10):
		print("==> Negative sampling, round "+str(i))
		train_neg_list, logo_bins, no_logo_list = generate_neg_pairs(class_list, dataset_path, patch_dir, clean_logo_path, 
			logo_rate, clean_logo_map, logo_bins, no_logo_list, logo_counts, no_logo_counts, class_map, fixed_logo_bins, 'train')
		print("==> Sampled negative samples: "+str(len(train_neg_list)))
	
	print("Done.")

	# class_map_txt_to_npy(os.path.join(root_dir, 'split/class_map.txt'), class_list)
	# class_map = read_class_map_pos(class_list)
	# print(class_map)
	# print(str(len(class_map)))
	