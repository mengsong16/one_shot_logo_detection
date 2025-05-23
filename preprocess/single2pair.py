import os
import sys
import math
import random
import shutil
import time
import numpy as np

sys.path.append("../")
from config import *

# Clear directory
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

# dump a list of strings to a file
def dump(l, file):
	print("Dump to " + file)
	with open(file, 'w') as f:
		for line in l:
			f.write(line)
		
	print("Done.")

# generate positive pairs
# format: patches/positives/_.jpg clean_logos/_.jpg label patch_class_id clean_logo_class_id
def generate_pos_pairs(pos_patch_list, dataset_dir):
	with open(pos_patch_list, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split(',') for x in lines]
	
	print('Read positive logo patches: '+str(len(lines)))

	
	logo_bins = {}
	pos_pairs = []

	for line in lines:
		pos_path = line[0]
		class_id = int(line[1])
		clean_logo_path = line[2]

		# paths
		pos_patch_path = os.path.join('patches', pos_path)
		clean_logo_path = os.path.join('clean_logos', clean_logo_path)
		# check correctness
		try:
			assert(os.path.exists(os.path.join(dataset_dir, pos_patch_path)))
			assert(os.path.exists(os.path.join(dataset_dir, clean_logo_path)))
		except AssertionError:
			print(os.path.join(dataset_dir, pos_patch_path))
			print(os.path.join(dataset_dir, clean_logo_path))


		# write positive pairs
		pos_pairs.append(pos_patch_path + ',' + clean_logo_path \
			+ ',' + str(1) + ',' + str(class_id) + ',' + str(class_id) + '\n')

		# add positive patch to logo bins
		if class_id not in logo_bins:
			logo_bins[class_id] = [pos_path]
		else:	
			logo_bins[class_id].append(pos_path)

	print("Generated "+str(len(pos_pairs))+" positive pairs.")

	return pos_pairs, logo_bins


# for each class, compute how many logo patches and no-logo patches do we need
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

# get clean logo instances except one given class
def get_clean_logo_instances_of_rest_classes(class_id, logo_bins, clean_logo_map, class_map):
	rest_ids = []
	for cls_id, paths in logo_bins.items():
		if cls_id != class_id:
			rest_ids.append(int(cls_id))

	rest_clean_logos = []
	for i in rest_ids:
		class_name = class_map[i]
		for l in clean_logo_map[i]:
			rest_clean_logos.append((class_name+"_"+str(l)+".jpg", i))

	return rest_clean_logos

# sample clean logo for the given brand, return a list of (clean_logo_instance_name, class_id)
def sample_clean_logo(class_id, class_map, clean_logo_map):
	ids = list(clean_logo_map[class_id])
	class_name = class_map[class_id]
	if len(ids) <= 0:
		print("Error: "+class_name+" has 0 clean logo instances.")
		return
	elif len(ids) == 1:
		return class_name+"_"+str(ids[0])+".jpg"	
	else:
		return class_name+"_"+str(np.random.choice(ids))+".jpg"		

# generate negative pairs which consists of a logo patch and a clean logo from other brands
# format: patches/positives/_.jpg clean_logos/_.jpg label patch_class_id clean_logo_class_id
def generate_neg_logo_pairs(clean_logo_map, class_map, logo_bins, logo_counts, dataset_dir):
	
	neg_logo_pairs = []
	for class_id, paths in logo_bins.items():
		# sample logo patches
		sample_n = logo_counts[class_id]
		assert len(paths) >= sample_n
		sampled_logo_patches = random.sample(paths, sample_n)
		# get clean logos from the rest classes
		rest_clean_logos = get_clean_logo_instances_of_rest_classes(class_id, logo_bins, clean_logo_map, class_map)
		# sample clean logos
		m = int(math.ceil(len(sampled_logo_patches) / float(len(rest_clean_logos)))) - 1
		sampled_clean_logos = rest_clean_logos * m
		sampled_clean_logos += random.sample(rest_clean_logos, len(sampled_logo_patches)-len(sampled_clean_logos))
		random.shuffle(sampled_clean_logos)
		if len(sampled_logo_patches) == len(sampled_clean_logos):
			name_pairs = zip(sampled_logo_patches, sampled_clean_logos)
		else:
			print("Error: the number of logo patches and clean logos should be equal.")	


		for logo_patch_name, (clean_logo_name, clean_logo_id) in name_pairs:

			patch_path = os.path.join('patches', logo_patch_name)
			clean_logo_path = os.path.join('clean_logos', clean_logo_name)

			# check correctness
			try:
				assert(os.path.exists(os.path.join(dataset_dir, patch_path)))
				assert(os.path.exists(os.path.join(dataset_dir, clean_logo_path)))
			except AssertionError:
				print(os.path.join(dataset_dir, patch_path))
				print(os.path.join(dataset_dir, clean_logo_path))

			neg_logo_pairs.append(patch_path + ',' + clean_logo_path \
			+ ',' + str(0) + ',' + str(class_id) + ',' + str(clean_logo_id) + '\n')

	
	
	print("Generated "+str(len(neg_logo_pairs))+" logo negative pairs.")
	return neg_logo_pairs

# generate negative pairs which consists of a non-logo patch and a clean logo from one brand
# format: patches/positives/_.jpg clean_logos/_.jpg label patch_class_id clean_logo_class_id 
def generate_neg_no_logo_pairs(neg_patch_list, clean_logo_map, class_map, no_logo_counts, dataset_dir):
	# read no logo patches
	with open(neg_patch_list, 'r') as f:
		lines = f.readlines()

	no_logo_list = [x.strip() for x in lines]
	
	
	# get the number of no-logo patches we need
	s = 0
	for class_id, n in no_logo_counts.items():
		s += n
	# uniformly sample s no-logo patches
	assert len(no_logo_list) >= s
	sampled_list = random.sample(no_logo_list, s)
	
	# assign them to each class
	neg_no_logo_pairs = []
	i = 0
	for class_id, n in no_logo_counts.items():
		class_id = int(class_id)
		for j in list(range(n)):
			neg_patch_path = os.path.join('patches', sampled_list[i])
			clean_logo_path = os.path.join('clean_logos', sample_clean_logo(class_id, class_map, clean_logo_map))

			# check correctness
			try:
				assert(os.path.exists(os.path.join(dataset_dir, neg_patch_path)))
				assert(os.path.exists(os.path.join(dataset_dir, clean_logo_path)))
			except AssertionError:
				print(os.path.join(dataset_dir, neg_patch_path))
				print(os.path.join(dataset_dir, clean_logo_path))
			# write to no logo neg pairs
			neg_no_logo_pairs.append(neg_patch_path + ',' + clean_logo_path \
			+ ',' + str(0) + ',' + str(0) + ',' + str(class_id) + '\n')
			i += 1

	print("Generated "+str(len(neg_no_logo_pairs))+" no-logo negative pairs.")
	
	return neg_no_logo_pairs


# generate pairs
# format: patches/positives/_.jpg clean_logos/_.jpg label patch_class_id clean_logo_class_id 
def generate_pairs(class_map_file, logo_rate, dataset_dir, pair_dir, patch_dir, clean_logo_dir, spec):

	# paths
	pos_patch_list = os.path.join(patch_dir, spec+'_pos.txt')
	neg_patch_list = os.path.join(patch_dir, spec+'_neg.txt')

	print("Generating pairs: " + spec)
	

	# read clean logo map
	clean_logo_map_file = os.path.join(clean_logo_dir, spec+'_clean_logo_map.npy')
	clean_logo_map = np.load(clean_logo_map_file).item()

	# read class map
	class_map = np.load(class_map_file).item()

	# sample positive pairs
	pos_pairs, logo_bins = generate_pos_pairs(pos_patch_list, dataset_dir)
	logo_counts, no_logo_counts = split_logo_bins(logo_bins, logo_rate)
	# sample negative pairs with logos
	if len(logo_counts) > 0:
		neg_logo_pairs = generate_neg_logo_pairs(clean_logo_map, class_map, logo_bins, logo_counts, dataset_dir)
	else:
		neg_logo_pairs = []
	# sample negative pairs without logos	
	if len(no_logo_counts) > 0:	
		neg_no_logo_pairs = generate_neg_no_logo_pairs(neg_patch_list, clean_logo_map, class_map, no_logo_counts, dataset_dir)
	else:
		neg_no_logo_pairs = []
	
	neg_pairs = neg_logo_pairs + neg_no_logo_pairs
	# dump to files
	if not os.path.exists(pair_dir): 
		os.makedirs(pair_dir)

	dump(pos_pairs, os.path.join(pair_dir, spec+'_pos.txt'))
	dump(neg_pairs, os.path.join(pair_dir, spec+'_neg.txt'))
	dump(pos_pairs+neg_pairs, os.path.join(pair_dir, spec+'_all.txt'))

	print("Generated "+str(len(pos_pairs)+len(neg_pairs))+" pairs in total.")
	print("Done.")
	print("=============================================================")

# main
if __name__ == '__main__':
	
	logo_rate = 0.5

	# redirect output 
	sys.stdout = open(os.path.join(fact_dir, 'fact_pair_gen.txt'), 'w')

	generate_pairs(class_map_file, logo_rate, train_dir, train_pair_dir, train_patch_dir, train_clean_logo_dir, 'train')
	generate_pairs(class_map_file, logo_rate, test_dir, test_pair_dir, test_patch_dir, test_clean_logo_dir, 'test_wo32')
	generate_pairs(class_map_file, logo_rate, test_dir, test_pair_dir, test_patch_dir, test_clean_logo_dir, 'test_w32')
	generate_pairs(class_map_file, logo_rate, test_dir, test_pair_dir, test_patch_dir, test_clean_logo_dir, 'test_seen')
	generate_pairs(class_map_file, logo_rate, val_dir, val_pair_dir, val_patch_dir, val_clean_logo_dir, 'val')
	generate_pairs(class_map_file, logo_rate, val_dir, val_pair_dir, val_patch_dir, val_clean_logo_dir, 'val_seen')


	
	
	
