import os
import sys
import math
import random
import shutil
import time
import numpy as np
import lmdb
import cv2
import caffe
from caffe.proto import caffe_pb2
from PIL import Image
import scipy.misc

sys.path.append("../")
from config import *

# for each class, gather logo patch paths
def build_logo_bins(patch_file):
	print("Constructing logo bins for "+patch_file)
	with open(patch_file, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split(',') for x in lines]
	
	N = len(lines)
	print('Read logo patches: '+str(N))

	#print("class_id: # pos patches")
	logo_bins = {}
	for line in lines:
		class_id = int(line[0])
		path = line[1]
		if class_id not in logo_bins:
			logo_bins[class_id] = [path]
		else:	
			logo_bins[class_id].append(path)

	# compute the number of classes and patches		
	n = 0
	c = 0
	for k, v in logo_bins.items():
		n += len(v)
		c += 1
		#print(str(k)+": "+str(len(v)))

	print("Classes: "+str(c))
	assert n == N
	print("Patches: "+str(n))

	return logo_bins, N

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

# generate positive pairs
def generate_pos_pairs(logo_bins):
	class_n = len(logo_bins)
	pos_pair_n = 0
	for class_id, patches in logo_bins.items():
		patch_n = len(patches)
		pos_pair_n += int(patch_n * (patch_n + 1) / 2.0)

	
	pos_pair_paths = [None] * pos_pair_n
	pos_pair_labels = [None] * pos_pair_n
	k = 0
	for class_id, patches in logo_bins.items():
		patch_n = len(patches)
		for i in list(range(patch_n)):
			for j in list(range(i, patch_n)):
				pos_pair_paths[k] = [patches[i], patches[j]]
				pos_pair_labels[k] = [class_id, class_id]
				k += 1

	assert k == pos_pair_n
	assert len(pos_pair_paths) == len(pos_pair_labels)
	
	print("Generated "+str(pos_pair_n)+" positive pairs.")

	result = list(zip(pos_pair_paths, pos_pair_labels))

	return [list(elem) for elem in result], pos_pair_n

# sample negative patches
def sample_neg_patches(neg_pair_required_n, rest_patches, class_id):
	if neg_pair_required_n < len(rest_patches):
		neg_patches = random.sample(rest_patches, neg_pair_required_n)
	elif neg_pair_required_n == len(rest_patches):
		neg_patches = rest_patches
	else:		
		m = int(math.ceil(neg_pair_required_n / float(len(rest_patches)))) - 1
		#print("Over sampling class "+str(class_id) +": [" +str(neg_pair_required_n) + "/" + str(len(rest_patches)) +"]")
		neg_patches = rest_patches * m
		neg_patches += random.sample(rest_patches, neg_pair_required_n-m*len(rest_patches))

	random.shuffle(neg_patches)	

	return neg_patches	

# map patch to class id
def get_patch_to_class_id_dictionary(logo_bins):
	reverse_dict = {}
	for class_id, patches in logo_bins.items():
		for patch in patches:
			reverse_dict[patch] = class_id

	print("Constructed reversed dictionary: "+str(len(reverse_dict)))
	return reverse_dict		

# get class ids for input patches
def get_class_ids(pair_paths, reverse_dict):
	pair_class_ids = [None] * len(pair_paths)
	i = 0
	for patch in pair_paths:
		pair_class_ids[i] = [reverse_dict[patch[0]], reverse_dict[patch[1]]]
		i += 1

	return pair_class_ids 	

# generate negative pairs
def generate_neg_pairs(logo_bins, reverse_dict, total_patch_n, pos_pair_n):
	neg_per_image_n = int(math.ceil(pos_pair_n/total_patch_n))
	neg_pair_n = neg_per_image_n * total_patch_n

	neg_pair_paths = []
	
	for class_id, patches in logo_bins.items():
		patch_n = len(patches)
		neg_pair_required_n = neg_per_image_n * patch_n
		rest_patches = get_rest_classes(class_id, logo_bins)
		this_class_patches = patches * neg_per_image_n
		neg_patches = sample_neg_patches(neg_pair_required_n, rest_patches, class_id)
		assert len(this_class_patches) == len(neg_patches)
		this_result = list(zip(this_class_patches, neg_patches))
		neg_pair_paths += [list(elem) for elem in this_result]
		
	neg_pair_labels = get_class_ids(neg_pair_paths, reverse_dict)

	print("Generated "+str(neg_pair_n)+" negative pairs.")

	result = list(zip(neg_pair_paths, neg_pair_labels))

	return [list(elem) for elem in result]

# generate pairs for training set
def generate_pairs(patch_file, batchsize):
	logo_bins, total_patch_n = build_logo_bins(patch_file)
	reverse_dict = get_patch_to_class_id_dictionary(logo_bins)
	pos_pairs, pos_pair_n = generate_pos_pairs(logo_bins)
	neg_pairs = generate_neg_pairs(logo_bins, reverse_dict, total_patch_n, pos_pair_n)
	total_n = len(pos_pairs) + len(neg_pairs)
	print("Generated "+str(total_n)+" pairs.")
	# for batch divisibility randomly delete the negative pairs
	pair_batch_size = int(batchsize / 2)
	num_to_delete = total_n - int(math.floor(total_n/pair_batch_size))*pair_batch_size
	print("Deleting "+str(num_to_delete)+" negative pairs...")
	for i in list(range(num_to_delete)):
		neg_pairs.pop(random.randrange(len(neg_pairs)))

	total_pairs = pos_pairs + neg_pairs
	random.shuffle(total_pairs)

	print("After deletion of negative pairs, there are "+str(len(total_pairs))+" pairs.")
	#print(total_pairs[0])

	return total_pairs

# serialize pairs
def serialize_pairs(pairs, batchsize):
	image_paths_serial = []
	labels_serial = []
	
	image_pairs, label_pairs = zip(*pairs)
	image_pairs = [list(ie) for ie in image_pairs]
	label_pairs = [list(le) for le in label_pairs]

	
	pair_batch_size = int(batchsize / 2)
	
	batch_n = int(len(image_pairs)/pair_batch_size)
	
	for i in list(range(len(image_pairs))):
		if i % pair_batch_size == 0:
			pairs = image_pairs[i:(i+pair_batch_size)]
			labels = label_pairs[i:(i+pair_batch_size)]
			image_a, image_b = zip(*pairs)
			label_a, label_b = zip(*labels)
			image_paths_serial += list(image_a)
			image_paths_serial += list(image_b)
			labels_serial += list(label_a)
			labels_serial += list(label_b)
			

	assert len(image_paths_serial) == 2*len(image_pairs)
	assert len(labels_serial) == 2*len(label_pairs)
	
	return image_paths_serial, labels_serial

# load image
# do not use io.imread(path), otherwise there will be error "'int' object is not subscriptable"
def default_image_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as image:
			# assume input image has 3 channels (not .png)
			# gray image will be converted to three chanel
			return image.convert('RGB')	

# read and resize image
def read_resize_image(image_file):
	# read in raw image as 8-bit bgr in range [0,255] with channels stored in BGR order
	image = default_image_loader(image_file)
	# resize into (256, 256)
	image = scipy.misc.imresize(image, (256,256)) 
	im_array = np.asarray(image, np.uint8)
	# array_to_datum: datum.channels, datum.height, datum.width
	# (height, width, channel) -> (channel, height, width)
	im_array = np.transpose(im_array, (2,0,1))

	return im_array
	
# write lmdb
def write_lmdb(lmdb_file, batch_size, image_paths_serial, labels_serial, dataset_dir):
	# create the lmdb file
	lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
	lmdb_txn = lmdb_env.begin(write=True)
	datum = caffe_pb2.Datum()

	# number of images
	image_n = len(image_paths_serial)

	# iterate over pair of (image, label)
	item_id = -1
	for i in list(range(image_n)):
		item_id += 1

		# get image path and label
		image_path = image_paths_serial[i]
		label = labels_serial[i]

		# prepare image and label
		image = read_resize_image(os.path.join(dataset_dir, image_path))
		
		# Converts a 3-dimensional array to datum
		datum = caffe.io.array_to_datum(image, label)
		keystr = '{:0>8d}'.format(item_id)
		# save in datum
		lmdb_txn.put( keystr, datum.SerializeToString() )

		# write current batch
		if (item_id+1) % batch_size == 0:
			lmdb_txn.commit()
			lmdb_txn = lmdb_env.begin(write=True)
			#print('Batch: ' + str((item_id+1)/batch_size))

		#break	
	
	# write last batch
	if (item_id+1) % batch_size != 0:
		lmdb_txn.commit()
		#print('Writing the last batch: '+ str(int(math.ceil(float(item_id + 1)/batch_size))))
	
	print("The number of batches: "+str(int(math.ceil(float(item_id + 1)/batch_size))))
	print("Serialization Done.")

# serialize test set
def serialize_test_set(patch_file, lmdb_file, dataset_dir, batch_size):
	print("Serialize "+lmdb_file+" ...")

	with open(patch_file, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split(',') for x in lines]
	lines = np.asarray(lines)
	class_ids = lines[:, 0]
	class_ids = [int(c) for c in class_ids]
	image_paths = lines[:, 1]

	print('Read logo patches: '+str(len(lines)))
	write_lmdb(lmdb_file, batch_size, image_paths, class_ids, dataset_dir)

	print("==================================================================")
	
# serialize training set
def serialize_training_set(patch_file, lmdb_file, dataset_dir, batch_size):
	print("Serialize "+lmdb_file+" ...")
	total_pairs = generate_pairs(patch_file, batch_size)
	image_paths_serial, labels_serial = serialize_pairs(total_pairs, batch_size)
	write_lmdb(lmdb_file, batch_size, image_paths_serial, labels_serial, dataset_dir)

	print("==================================================================")
	
# main
if __name__ == '__main__':
	# redirect output 
	sys.stdout = open(os.path.join(fact_dir, 'fact_serialization_for_embedding.txt'), 'w')

	batch_size = 128
	print("Batch size: "+str(batch_size))
	print("==================================================================")
	
	#serialize_training_set(os.path.join(train_lmdb_dir, 'train_patch_list.txt'), os.path.join(train_lmdb_dir, 'train.lmdb'), train_dir, batch_size)
	
	serialize_test_set(os.path.join(test_lmdb_dir, 'test_w32_patch_list.txt'), os.path.join(test_lmdb_dir, 'test_w32_patch.lmdb'), test_dir, batch_size)
	serialize_test_set(os.path.join(test_lmdb_dir, 'test_wo32_patch_list.txt'), os.path.join(test_lmdb_dir, 'test_wo32_patch.lmdb'), test_dir, batch_size)
	serialize_test_set(os.path.join(test_lmdb_dir, 'test_seen_patch_list.txt'), os.path.join(test_lmdb_dir, 'test_seen_patch.lmdb'), test_dir, batch_size)
	serialize_test_set(os.path.join(val_lmdb_dir, 'val_seen_patch_list.txt'), os.path.join(val_lmdb_dir, 'val_seen_patch.lmdb'), val_dir, batch_size)
	serialize_test_set(os.path.join(val_lmdb_dir, 'val_patch_list.txt'), os.path.join(val_lmdb_dir, 'val_patch.lmdb'), val_dir, batch_size)
	
	serialize_test_set(os.path.join(csv_cleanlogo_dir, 'test_w32_cleanlogos.txt'), os.path.join(test_lmdb_dir, 'test_w32_cleanlogo.lmdb'), test_clean_logo_dir, batch_size)
	serialize_test_set(os.path.join(csv_cleanlogo_dir, 'test_wo32_cleanlogos.txt'), os.path.join(test_lmdb_dir, 'test_wo32_cleanlogo.lmdb'), test_clean_logo_dir, batch_size)
	serialize_test_set(os.path.join(csv_cleanlogo_dir, 'test_seen_cleanlogos.txt'), os.path.join(test_lmdb_dir, 'test_seen_cleanlogo.lmdb'), test_clean_logo_dir, batch_size)
	serialize_test_set(os.path.join(csv_cleanlogo_dir, 'val_seen_cleanlogos.txt'), os.path.join(val_lmdb_dir, 'val_seen_cleanlogo.lmdb'), val_clean_logo_dir, batch_size)
	serialize_test_set(os.path.join(csv_cleanlogo_dir, 'val_cleanlogos.txt'), os.path.join(val_lmdb_dir, 'val_cleanlogo.lmdb'), val_clean_logo_dir, batch_size)
	
	