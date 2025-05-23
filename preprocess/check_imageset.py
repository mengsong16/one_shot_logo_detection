import os
import sys
import math
import random
import shutil
import time
import numpy as np
import scipy.misc
from PIL import Image

# read all images in a directory
def list_all_images(directory,  ext):
	#print('Checking '+directory)
	files = []
	for file in os.listdir(directory):
		if file.endswith(ext):
			files.append(os.path.join(directory, file))
		else:
			print(os.path.join(directory, file))	

	#print('Checking completed!')
	return files


# main function 
if __name__ == '__main__':

	dataset_dir = '/data/flickr_100m_logo_dataset/flickr_100m_logo'	
	train_image_dir = os.path.join(dataset_dir, 'train/images')
	test_image_dir = os.path.join(dataset_dir, 'test/images')
	val_image_dir = os.path.join(dataset_dir, 'val/images')


	# redirect output 
	sys.stdout = open('/data/flickr_100m_logo_dataset/fact/fact_image_set.txt', 'w')

	train_imgs = list_all_images(train_image_dir,  '.jpg')
	print("training images: "+str(len(train_imgs)))
	test_imgs = list_all_images(test_image_dir,  '.jpg')
	print("test images: "+str(len(test_imgs)))
	val_imgs = list_all_images(val_image_dir,  '.jpg')
	print("validation images: "+str(len(val_imgs)))

	

