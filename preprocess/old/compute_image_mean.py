import os
import sys
import math
import random
import shutil
import time
import numpy as np
import scipy.misc

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

def read_image_list(image_list):
	with open(image_list, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	print('Read images: '+str(np.shape(lines)[0]))

	return lines[:,2]

# write computed mean value in the given file
def write_mean_result(result, file_dir):
	with open(os.path.join(file_dir), 'w') as f:
		f.write(str(result[0])+' '+str(result[1])+' '+str(result[2])+'\n')

# compute mean for one image
def compute_mean_image(file_dir):
	r = np.zeros(3)
	im = scipy.misc.imread(file_dir).astype(float) # shape of im is (h,w,3) in order of RGB
	h, w = im.shape[:2]
	# do it in two steps to prevent overflow
	r = np.sum(im, axis=0)
	r /= float(h)
	r = np.sum(r, axis=0)
	r /= float(w)
	return r

# compute mean value of training images
def compute_mean(image_dir, train_list, mean_file):

	# read training images
	paths = read_image_list(train_list)
	n = len(paths)
	# Start computing 
	print('Start computing mean image for training set...')
	res = np.zeros(3)
	i = 0
	for path in paths:
		res += compute_mean_image(os.path.join(image_dir,path))
		i += 1

		if i % 10 == 0:
			print(str(i)+ ' Done.')

	res /= float(n)

	write_mean_result(res, mean_file)
	print('Done: ', res)

	return

# main function  
if __name__ == '__main__':
	train_list = os.path.join(dataset_path, 'split/train.txt')
	mean_file = os.path.join(dataset_path, 'split/mean.txt')

	print('Start computing mean images for the training set...')
	compute_mean(image_dir, train_list, mean_file)

