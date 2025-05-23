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


# generate pairs for training set
def generate_pairs(patch_file, batchsize):
	# read pairs
	total_pairs = []
	image_pairs = []
	label_pairs = []
	with open(patch_file, 'r') as pinf:
		for line in pinf:
			items = line.strip().split(',')
			image_pairs.append([items[0],items[1]])
			label_pairs.append([int(items[3]),int(items[4])])

	total_pairs = zip(image_pairs, label_pairs)		
	total_n = len(total_pairs)
	print("Read "+str(total_n)+" pairs.")
	# for batch divisibility randomly delete the negative pairs
	pair_batch_size = int(batchsize / 2)
	num_to_delete = total_n - int(math.floor(total_n/pair_batch_size))*pair_batch_size
	print("Deleting "+str(num_to_delete)+" negative pairs...")
	for i in list(range(num_to_delete)):
		total_pairs.pop(random.randrange(total_n))

	# shuffle	
	random.shuffle(total_pairs)

	print("After deletion of negative pairs, there are "+str(len(total_pairs))+" pairs.")

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
	
def count_lmdb_item_num(LMDB_FILENAME):
	lmdb_env = lmdb.open(LMDB_FILENAME)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	num_imgs = 0
	for key in lmdb_cursor:
	    num_imgs += 1

	return num_imgs

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

# serialize test logo patches
def serialize_test_logo_patches(pos_patch_file, lmdb_file, dataset_dir, batch_size):
	print("Serialize "+lmdb_file+" ...")

	# read positive patches
	image_paths = []
	class_ids = []
	with open(pos_patch_file, 'r') as pinf:
		for line in pinf:
			items = line.strip().split(',')
			class_ids.append(int(items[1]))
			image_paths.append(items[0])


	print('Read logo patches: '+str(len(image_paths)))
	write_lmdb(lmdb_file, batch_size, image_paths, class_ids, dataset_dir)

	# check number of images written into lmdb file
	print('Written number of images: '+str(count_lmdb_item_num(lmdb_file)))

	print("==================================================================")

# serialize test logo patches
def serialize_test_cleanlogospos(patch_file, lmdb_file, dataset_dir, batch_size):
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

	# check number of images written into lmdb file
	print('Written number of images: '+str(count_lmdb_item_num(lmdb_file)))

	print("==================================================================")

# serialize training set
def serialize_training_set(pair_file, lmdb_file, dataset_dir, batch_size):
	print("Serialize "+lmdb_file+" ...")
	total_pairs = generate_pairs(pair_file, batch_size)
	#print(total_pairs[0:2])
	image_paths_serial, labels_serial = serialize_pairs(total_pairs, batch_size)
	#print(image_paths_serial[0:2])
	#print(image_paths_serial[64:66])
	#print(labels_serial[0:2])
	#print(labels_serial[64:66])
	write_lmdb(lmdb_file, batch_size, image_paths_serial, labels_serial, dataset_dir)

	# check number of images written into lmdb file
	print('Written number of images: '+str(count_lmdb_item_num(lmdb_file)))

	print("==================================================================")
	


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

# main
if __name__ == '__main__':
	# redirect output 
	sys.stdout = open(os.path.join(fact_dir, 'fact_serialization_for_embedding.txt'), 'w')

	batch_size = 128


	
	# clear lmdb folders
	clear_dir(test_lmdb_dir)
	clear_dir(val_lmdb_dir)
	clear_dir(train_lmdb_dir)
	
	print("Batch size: "+str(batch_size))
	print("==================================================================")
	
	
	serialize_test_logo_patches(os.path.join(test_patch_dir, 'test_w32_pos.txt'), os.path.join(test_lmdb_dir, 'test_w32_patch.lmdb'), test_patch_dir, batch_size)
	serialize_test_logo_patches(os.path.join(test_patch_dir, 'test_wo32_pos.txt'), os.path.join(test_lmdb_dir, 'test_wo32_patch.lmdb'), test_patch_dir, batch_size)
	serialize_test_logo_patches(os.path.join(test_patch_dir, 'test_seen_pos.txt'), os.path.join(test_lmdb_dir, 'test_seen_patch.lmdb'), test_patch_dir, batch_size)
	serialize_test_logo_patches(os.path.join(val_patch_dir, 'val_pos.txt'), os.path.join(val_lmdb_dir, 'val_patch.lmdb'), val_patch_dir, batch_size)
	serialize_test_logo_patches(os.path.join(val_patch_dir, 'val_seen_pos.txt'), os.path.join(val_lmdb_dir, 'val_seen_patch.lmdb'), val_patch_dir, batch_size)
	
	serialize_test_cleanlogospos(os.path.join(csv_cleanlogo_dir, 'test_w32_cleanlogos.txt'), os.path.join(test_lmdb_dir, 'test_w32_cleanlogo.lmdb'), test_clean_logo_dir, batch_size)
	serialize_test_cleanlogospos(os.path.join(csv_cleanlogo_dir, 'test_wo32_cleanlogos.txt'), os.path.join(test_lmdb_dir, 'test_wo32_cleanlogo.lmdb'), test_clean_logo_dir, batch_size)
	serialize_test_cleanlogospos(os.path.join(csv_cleanlogo_dir, 'test_seen_cleanlogos.txt'), os.path.join(test_lmdb_dir, 'test_seen_cleanlogo.lmdb'), test_clean_logo_dir, batch_size)
	serialize_test_cleanlogospos(os.path.join(csv_cleanlogo_dir, 'val_seen_cleanlogos.txt'), os.path.join(val_lmdb_dir, 'val_seen_cleanlogo.lmdb'), val_clean_logo_dir, batch_size)
	serialize_test_cleanlogospos(os.path.join(csv_cleanlogo_dir, 'val_cleanlogos.txt'), os.path.join(val_lmdb_dir, 'val_cleanlogo.lmdb'), val_clean_logo_dir, batch_size)
	
	serialize_training_set(os.path.join(train_pair_dir, 'train_all.txt'), os.path.join(train_lmdb_dir, 'train.lmdb'), train_dir, batch_size)
	

	
	