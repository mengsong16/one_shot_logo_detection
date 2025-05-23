import os
import sys
import math
import random
import shutil
import time
import numpy as np
import sys
import json
from PIL import Image, ImageDraw
from random import randint
import datetime
import pandas as pd

sys.path.append("../")
from config import *

# do not use io.imread(path), otherwise there will be error "'int' object is not subscriptable"
def default_image_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as image:
			# assume input image has 3 channels (not .png)
			# gray image will be converted to three chanel
			#img = np.array(img)/255.0 for imshow
			return image.convert('RGB')	

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

def _draw_rectangle(drawing, xmin, ymin, xmax, ymax, color, width=1):
	coordinates = ((xmin, ymin), (xmax, ymax))
	for i in range(width):
		rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
		rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
		drawing.rectangle((rect_start, rect_end), outline=color)

# compute iou
def iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
	x1 = max(xmin1, xmin2)
	y1 = max(ymin1, ymin2)
	x2 = min(xmax1, xmax2)
	y2 = min(ymax1, ymax2)

	w = x2 - x1 + 1
	h = y2 - y1 + 1

	if w <= 0 or h <=0:
		iu = 0
	else:	
		# intersection over union overlap
		inter = w * h
		a_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
		b_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
		iu = inter / float(a_area + b_area - inter)

	return iu	


# center may to left move than the actual value, and the size will be larger by 1
# assume inputs are integers
def from_corner_to_center(g_xmin, g_ymin, g_xmax, g_ymax):
	g_half_width = int((g_xmax - g_xmin) / 2)
	g_half_height = int((g_ymax - g_ymin) / 2)

	g_x_center = g_xmin + g_half_width
	g_y_center = g_ymin + g_half_height
	

	return g_x_center, g_y_center, g_half_width, g_half_height

def from_center_to_corner(g_x_center, g_y_center, g_half_width, g_half_height):
	g_xmin = g_x_center - g_half_width
	if g_xmin < 0:
		g_xmin = 0
	g_xmax = g_x_center + g_half_width
	g_ymin = g_y_center - g_half_height
	if g_ymin < 0:
		g_ymin = 0
	g_ymax = g_y_center + g_half_height

	return int(g_xmin), int(g_ymin), int(g_xmax), int(g_ymax)

# uniformly sample patch on an image
def uniform_crop(width, height, n_min_size, n_max_size, g_xmin, g_ymin, g_xmax, g_ymax, iou_th):
	
	# generate one patch that satisfying size requirement
	while True:
		# sample mins
		p_xmin = random.randint(0, width-1)
		p_ymin = random.randint(0, height-1)

		p_min_xmax = int(n_min_size + p_xmin - 1)
		p_min_ymax = int(n_min_size + p_ymin - 1)
		
		# p_xmin or p_ymin is not valid
		if p_min_xmax > width-1 or p_min_ymax > height-1:
			continue
		
		# sample maxs
		p_xmax = random.randint(p_min_xmax, width-1)
		p_ymax = random.randint(p_min_ymax, height-1)

		# p_xmax or p_ymax is not valid
		if (p_xmax-p_xmin+1) > n_max_size or (p_ymax-p_ymin+1) > n_max_size:
			continue

		break

	# classify the patch as positive or negative 	
	iou_v = iou(g_xmin, g_ymin, g_xmax, g_ymax, p_xmin, p_ymin, p_xmax, p_ymax)

	if iou_v >= iou_th:
		flag = True
	else:
		flag = False

	return p_xmin, p_ymin, p_xmax, p_ymax, flag	

def x_boundary_check(a, width):	
	if a < 0 or a > width-1:
		return False
	else:
		return True	

def y_boundary_check(a, height):	
	if a < 0 or a > height-1:
		return False
	else:
		return True		

# sample a patch around the ground truth bbox
def around_bbox_crop(width, height, p_min_size, g_xmin, g_ymin, g_xmax, g_ymax, iou_th):
	# get center-size representation of ground truth bbox
	g_x_center, g_y_center, g_half_width, g_half_height = from_corner_to_center(g_xmin, g_ymin, g_xmax, g_ymax)
	
	# generate one patch that satisfying size requirement
	while True:
		p_x_center = int(np.random.normal(g_x_center, g_half_width/3))
		p_y_center = int(np.random.normal(g_y_center, g_half_height/3))
		p_half_width = int(np.random.normal(g_half_width, width/6))
		p_half_height = int(np.random.normal(g_half_height, height/6))
		
		# size is not valid
		if p_half_width < p_min_size / 2 or p_half_height < p_min_size / 2 or p_half_width >= width / 2 or p_half_height >= height / 2:
			continue
			

		p_xmin, p_ymin, p_xmax, p_ymax = from_center_to_corner(p_x_center, p_y_center, p_half_width, p_half_height)	

		# coordinates are not valid
		if x_boundary_check(p_xmin, width) and x_boundary_check(p_xmax, width) and y_boundary_check(p_ymin, height) and y_boundary_check(p_ymax, height):
			break

	
	#print((p_xmin, p_ymin, p_xmax, p_ymax))
	# classify the patch as positive or negative 	
	iou_v = iou(g_xmin, g_ymin, g_xmax, g_ymax, p_xmin, p_ymin, p_xmax, p_ymax)

	#print(iou_v)

	if iou_v >= iou_th:
		flag = True
	else:
		flag = False

	return p_xmin, p_ymin, p_xmax, p_ymax, flag	

# index patch from 0
def generate_patches(csv_file, image_dir, patch_dir, neg_n, iou_th, spec):


	# build patch directories, don't clear it since different csv file may ask to enter the same patch folder
	if not os.path.exists(patch_dir): 
		os.makedirs(patch_dir)
	'''	
	else:
		clear_dir(patch_dir)
	'''	
	
	pos_patch_dir = os.path.join(patch_dir, 'positives')	
	if not os.path.exists(pos_patch_dir): 
		os.makedirs(pos_patch_dir)

	neg_patch_dir = os.path.join(patch_dir, 'negatives')	
	if not os.path.exists(neg_patch_dir): 
		os.makedirs(neg_patch_dir)
			
	# read csv	
	df = pd.read_csv(csv_file, usecols=['image_base_name', 'detections_id', 'detections_coordinates', 'brand_instance_logo'])
	# for each image record how many logo detection groudtruth does it have
	logo_dections = {}
	nnp = 0
	nn = 0
	with open(os.path.join(patch_dir, spec+'_pos.txt'), 'w') as pf:
		with open(os.path.join(patch_dir, spec+'_neg.txt'), 'w') as nf:
			# itererate over each image
			for i in list(range(len(df))):
				l = df.iloc[i,:]
				class_id = int(l['detections_id'])
				image_name = str(l['image_base_name'])
				coords = str(l['detections_coordinates'])
				clean_logo = os.path.splitext(str(l['brand_instance_logo']))[0]+'.jpg'
				
				# read image
				image =  default_image_loader(os.path.join(image_dir, image_name))
				# get min patch size
				width, height = image.size
				n_max_size = 0.25*max(width, height)
				n_min_size = 0.1*min(width, height)
				# read coordinates
				g_xmin, g_ymin, g_xmax, g_ymax = get_rectangle(coords)
				# update dictionary
				if image_name in logo_dections:
					logo_dections[image_name] += 1
				else:
					logo_dections[image_name] = 0	
				# crop and save positive patch
				img_p = image.crop((g_xmin, g_ymin, g_xmax, g_ymax))
				name_p = os.path.splitext(image_name)[0] + '_' + str(logo_dections[image_name]) + os.path.splitext(image_name)[1]
				path_p = os.path.join(pos_patch_dir, name_p)
				img_p.save(path_p)
				pf.write(os.path.join('positives',name_p)+ "," + str(class_id) + "," + clean_logo + '\n')
				pf.flush()
				nnp += 1
				
				# negative sample
				if logo_dections[image_name] == 0:
					neg_patches = set()
					while len(neg_patches) < neg_n : 
						p_xmin, p_ymin, p_xmax, p_ymax, flag = uniform_crop(width, height, n_min_size, n_max_size, g_xmin, g_ymin, g_xmax, g_ymax, iou_th)
						if flag == False:
							neg_patches.add((p_xmin, p_ymin, p_xmax, p_ymax))

					# crop and save negative samples
					k = 0
					for neg_p in neg_patches:
						# crop and save image
						img_p = image.crop(neg_p)
						name_p = os.path.splitext(image_name)[0] + '_' + str(k) + os.path.splitext(image_name)[1]
						path_p = os.path.join(neg_patch_dir, name_p)
						img_p.save(path_p)
						nf.write(os.path.join('negatives',name_p) + '\n')
						nf.flush()
						k += 1
						nn += 1
					

				# verbose
				#if (i+1) % 50 == 0:
				#	print(str(i+1) + " Done.")

				#break
		
	# save logo dection count dictionary
	np.save(os.path.join(patch_dir, spec+'_logo_detect_count.npy'), logo_dections)

	print("Done with "+str(len(df))+" detection instances: "+spec)
	print("Generated "+str(nnp)+ " / "+str(len(df))+" positive patches")
	print("Generated "+str(nn)+ " / "+str(len(logo_dections)*neg_n)+" negative patches")
	print("Generated "+str(nnp+nn)+" patches")
	print("==========================================")

# find the largest rectangle bounding box
def get_rectangle(coords):
	x1,y1,x2,y2,x3,y3,x4,y4 = coords.split(",")
	x1 = float(x1)
	x2 = float(x2)
	x3 = float(x3)
	x4 = float(x4)
	y1 = float(y1)
	y2 = float(y2)
	y3 = float(y3)
	y4 = float(y4)
	left = min([x1,x2,x3,x4])
	right = max([x1,x2,x3,x4])
	upper = min([y1,y2,y3,y4])
	bottom = max([y1,y2,y3,y4])
	return left,upper,right,bottom

# main
if __name__ == '__main__':

	# how many no-logo patches per image
	neg_n = 3
	iou_th = 0.75

	# redirect output 
	sys.stdout = open(os.path.join(fact_dir, 'fact_patch_gen.txt'), 'w')

	start = time.time()
	
	generate_patches(train_csv, train_image_dir, train_patch_dir, neg_n, iou_th, 'train')
	generate_patches(test_wo32_csv, test_image_dir, test_patch_dir, neg_n, iou_th, 'test_wo32')
	generate_patches(test_w32_csv, test_image_dir, test_patch_dir, neg_n, iou_th, 'test_w32')
	generate_patches(test_seen_csv, test_image_dir, test_patch_dir, neg_n, iou_th, 'test_seen')
	generate_patches(val_csv, val_image_dir, val_patch_dir, neg_n, iou_th, 'val')
	generate_patches(val_seen_csv, val_image_dir, val_patch_dir, neg_n, iou_th, 'val_seen')

	print("Time: "+str(datetime.timedelta(seconds=(time.time()-start))))
	
	



	