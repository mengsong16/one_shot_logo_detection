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

# read all files in a directory
def list_all_files(directory,  ext):
	files = []
	for file in os.listdir(directory):
		if file.endswith(ext):
			files.append(os.path.join(directory, file))

	return files		

# generate training set and test set, and generate image ids (image id: starting from 1)
# to run the code, ensure that image ids start from training set 
# format:  image_id class_id path

# also save the class split as txt
def gen_train_test(image_dir, output_dir, class_map, train, test, ext):
	i = 1 
	# training set
	train_file = os.path.join(output_dir, 'train.txt')
	train_class_file = os.path.join(output_dir, 'train_class.txt')
	
	n_train = 0 
	with open(train_file, 'w') as train_f: 
		with open(train_class_file, 'w') as train_class_f:
			for train_id in train:
				folder_name = class_map[train_id]
				train_class_f.write(str(train_id) + ' ' + folder_name + '\n')
				folder_full = os.path.join(image_dir, folder_name)
				for file in os.listdir(folder_full):
					if file.endswith(ext):
						train_f.write(str(i) + ' ' + str(train_id) + ' ' + os.path.join(folder_name, file) + '\n')
						n_train += 1 
						i += 1
	# test set
	n_test = 0
	test_file = os.path.join(output_dir, 'test.txt')
	test_class_file = os.path.join(output_dir, 'test_class.txt')
	test_class_f = open(test_class_file, 'w')
	with open(test_file, 'w') as test_f:
		with open(test_class_file, 'w') as test_class_f:
			for test_id in test:
				folder_name = class_map[test_id]
				test_class_f.write(str(test_id) + ' ' + folder_name + '\n')
				folder_full = os.path.join(image_dir, folder_name)
				for file in os.listdir(folder_full):
					if file.endswith(ext):
						test_f.write(str(i) + ' ' + str(test_id) + ' ' + os.path.join(folder_name, file) + '\n')
						n_test += 1
						i += 1				

	return n_train, n_test					


# get mapping between class name and class index (class id: starting from 1)
# save both txt and npy 
def get_class_map(image_dir, output_dir):
	class_map = {}
	with open(os.path.join(output_dir, 'class_map.txt'), 'w') as f:
		i = 1
		for folder in os.listdir(image_dir):
			folder_full = os.path.join(image_dir, folder)
			if os.path.isdir(folder_full):
				class_map[i] = folder
				f.write(str(i)+' '+class_map[i]+ '\n')
				i += 1

	np.save(os.path.join(output_dir, 'class_map.npy'), class_map)

	return class_map
				

def split_class(class_map, train_rate):
	total_n = len(class_map)

	train_n = int(math.floor(total_n * train_rate))
	# class id starts from 1
	p = list(range(1,total_n+1))
	random.shuffle(p)
	train = p[:train_n]
	test = p[train_n:]
	
	return train, test


# main
if __name__ == '__main__':
	
	output_dir = os.path.join(dataset_path, 'split')
	train_rate = 0.9
	ext = '.jpg'
	
	
	# creat or clear output dir
	if not os.path.exists(output_dir): 
		os.makedirs(output_dir)
	else:	
		clear_dir(output_dir)

	# redirect output 
	sys.stdout = open(os.path.join(output_dir, 'fact.txt'), 'w')
	

	# Start
	class_map = get_class_map(image_dir, output_dir)

	
	
	train_classes, test_classes = split_class(class_map, train_rate)
	n_train, n_test	 = gen_train_test(image_dir, output_dir, class_map, train_classes, test_classes, ext)
	print('Training classes: ' + str(len(train_classes)))
	print('Training images: ' + str(n_train))
	print('Test classes: ' + str(len(test_classes)))
	print('Test images: ' + str(n_test))
	print('Total classes:' + str(len(class_map)))
	print('Total images: ' + str(n_train+n_test))

	# check whether the number of images in training and test sets are approximately satisfying train_rate

	

	