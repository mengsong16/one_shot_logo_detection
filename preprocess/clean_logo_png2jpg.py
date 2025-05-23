import os
import sys
import math
import random
import shutil
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append("../")
from config import *

def read_mean_image(path):
	R = 0
	G = 0
	B = 0
	with open(path, 'r') as rf:
		line = rf.readline()
		R,G,B = line.strip().split()

	return int(float(R)), int(float(G)), int(float(B))

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


def png2jpg(source_file, target_file, mean_img):
	with open(source_file, 'rb') as f:
		with Image.open(f) as img:
			background =  Image.new('RGB', img.size, mean_img)
			img = Image.composite(img, background, img) # image1, image2, mask(could be RGBA, L ,1)
			img = img.convert('RGB')
			img.save(target_file, "JPEG")


def convert_clean_logos(png_logo_dir, jpg_logo_dir, mean_img):
	print('Start converting clean logos in '+png_logo_dir)

	if not os.path.exists(jpg_logo_dir): 
		os.makedirs(jpg_logo_dir)
	else:	
		clear_dir(jpg_logo_dir)

	# convert png to jpg
	j = 0
	
	# convert all files in source_folder from png to jpg
	# and save in target_folder
	for file in os.listdir(png_logo_dir):
		if file.endswith(".png"):
			source_file = os.path.join(png_logo_dir, file)
			target_file = os.path.join(jpg_logo_dir, os.path.splitext(file)[0] + '.jpg')
			png2jpg(source_file, target_file, mean_img)
			j += 1
	
	print("Converted " + str(j) + " logos.")	

	 
# main
if __name__ == '__main__':

	# redirect output 
	sys.stdout = open(os.path.join(fact_dir, 'fact_cleanlogo.txt'), 'w')
	# read mean image
	R,G,B = read_mean_image(mean_file)
	# train
	convert_clean_logos(os.path.join(train_dir, "logos"), train_clean_logo_dir, (R,G,B))
	# test
	convert_clean_logos(os.path.join(test_dir, "logos"), test_clean_logo_dir, (R,G,B))
	# validation
	convert_clean_logos(os.path.join(val_dir, "logos"), val_clean_logo_dir, (R,G,B))
	print('Done.')