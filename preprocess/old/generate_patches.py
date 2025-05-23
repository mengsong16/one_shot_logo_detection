import os
import sys
import math
import random
import shutil
import time
import numpy as np
import sys
import json
import xmltodict
from PIL import Image, ImageDraw
from random import randint
import datetime

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

	return lines[:,1], lines[:,2]

def read_class_list(class_list):
	with open(class_list, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	print('Read classes: '+str(np.shape(lines)[0]))

	return lines[:,0], lines[:,1]	

def _draw_rectangle(drawing, xmin, ymin, xmax, ymax, color, width=1):
	coordinates = ((xmin, ymin), (xmax, ymax))
	for i in range(width):
		rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
		rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
		drawing.rectangle((rect_start, rect_end), outline=color)

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

def build_dir_structure(patch_dir, class_list):
	class_map = read_class_map(class_list)
	
	for k,v in class_map.items():
		path = os.path.join(patch_dir, v)
		if not os.path.exists(path): 
			os.makedirs(path)
		else:	
			clear_dir(path)

	return class_map
			
# include no logo, test classes and training classes
def read_class_map(class_list):
	class_map = np.load(class_list).item()
	# map no_logo to 0
	# map other classes to positive integers 
	class_map[0] = 'no_logo'
	return class_map

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
def uniform_crop(width, height, p_min_size, g_xmin, g_ymin, g_xmax, g_ymax, iou_th):
	
	# generate one patch that satisfying size requirement
	while True:
		p_xmin = random.randint(0, width-1)
		p_ymin = random.randint(0, height-1)

		p_min_xmax = int(p_min_size + p_xmin - 1)
		p_min_ymax = int(p_min_size + p_ymin - 1)
		
		# p_xmin or p_ymin is not valid
		if p_min_xmax > width-1 or p_min_ymax > height-1:
			continue
		else:
			p_xmax = random.randint(p_min_xmax, width-1)
			p_ymax = random.randint(p_min_ymax, height-1)
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

def generate_patches(image_list, image_dir, patch_dir, class_map, pos_n, neg_n, p_min_size, iou_th, spec, start_ind):

	class_ids, image_paths = read_image_list(image_list)
	N = np.shape(image_paths)[0]

	n = start_ind # index patch from start_ind
	with open(os.path.join(patch_dir, spec+'_pos_list.txt'), 'w') as pf:
		with open(os.path.join(patch_dir, spec+'_neg_list.txt'), 'w') as nf:

			# itererate over each image
			for i in list(range(N)):
				g_xmin, g_ymin, g_xmax, g_ymax = read_bbox_xml(image_paths[i], image_dir)
				image = Image.open(os.path.join(image_dir, image_paths[i]))
				class_id = int(class_ids[i])
				class_name = class_map[class_id]
				image_name = os.path.basename(image_paths[i])
				width, height = image.size
				n_min_size = 0.25*min(width, height)
				pos_patches = set()
				neg_patches = set()
				# ground truth bbox is always the first positive example
				pos_patches.add((g_xmin, g_ymin, g_xmax, g_ymax))
				
				# iterate over each patch
				# positive sample: data augmentation for the ground truth bbox
				#start = time.time()
				while len(pos_patches) < pos_n :
					p_xmin, p_ymin, p_xmax, p_ymax, flag = around_bbox_crop(width, height, p_min_size, g_xmin, g_ymin, g_xmax, g_ymax, iou_th)
					if flag == True:
						pos_patches.add((p_xmin, p_ymin, p_xmax, p_ymax))
				#print("Time: "+str(time.time()-start))	
				# negative sample	
				#start = time.time()
				while len(neg_patches) < neg_n : 
					p_xmin, p_ymin, p_xmax, p_ymax, flag = uniform_crop(width, height, n_min_size, g_xmin, g_ymin, g_xmax, g_ymax, iou_th)
					if flag == False:
						neg_patches.add((p_xmin, p_ymin, p_xmax, p_ymax))

				#print("Time: "+str(time.time()-start))
				# save the patches for current image
				k = 1  # index sub index from 1 
				# pos
				for pos_p in pos_patches:
					# crop and save image
					img_p = image.crop(pos_p)
					name_p = os.path.splitext(image_name)[0] + '_' + str(k) + os.path.splitext(image_name)[1]
					path_p = os.path.join(class_name, name_p)
					full_path_p = os.path.join(patch_dir, path_p)
					img_p.save(full_path_p)
					k += 1
					# write to the list file
					# format: patch_ids, patch_path_list
					pf.write(str(n) + ' ' + str(class_id) + ' ' + path_p + '\n')
					n += 1

				'''
				if (g_xmin, g_ymin, g_xmax, g_ymax) in pos_patches:
					print((g_xmin, g_ymin, g_xmax, g_ymax))	
					print(g_xmax-g_xmin, g_ymax-g_ymin)
				else:
					print("GT Does not exist!")	
				'''

				# neg
				for neg_p in neg_patches:
					# crop and save image
					img_p = image.crop(neg_p)
					name_p = class_name + '_' + os.path.splitext(image_name)[0] + '_' + str(k) + os.path.splitext(image_name)[1]
					path_p = os.path.join("no_logo", name_p)
					full_path_p = os.path.join(patch_dir, path_p)
					img_p.save(full_path_p)
					k += 1
					# write to the list file
					# format: image_ids, class_ids, path_list
					nf.write(str(n) + ' ' + path_p + '\n')
					n += 1

				# verbose
				if (i+1) % 50 == 0:
					print(str(i+1) + " Done.")

				#break
		

	print("Done with "+str(N)+" images.")
	print("Generated "+str(n-start_ind)+" image patches.")

	return n

def read_bbox_xml(image_path, image_dir):
	xml_file = os.path.splitext(image_path)[0] + '.xml'
	xml_path = os.path.join(image_dir, xml_file)
	if os.path.isfile(xml_path):
		data = xmltodict.parse(open(xml_path).read())
		objs = data['annotation']['object']
		if type(objs) is list:
			# assume single bbox (Could work for logosc300)
			obj = objs[0]
			bbox = obj['bndbox']
			xmin = int(bbox['xmin'])
			ymin = int(bbox['ymin'])
			xmax = int(bbox['xmax'])
			ymax = int(bbox['ymax'])
		else:
			bbox = objs['bndbox']
			xmin = int(bbox['xmin'])
			ymin = int(bbox['ymin'])
			xmax = int(bbox['xmax'])
			ymax = int(bbox['ymax'])

		return xmin, ymin, xmax, ymax	
	else:	
		print("Error: "+xml_path+" DOES NOT exist.")
		return 0

# i: image number
# c: class number
def check_all_images(image_dir, ext):
	i = 0
	c = 0
	
	for folder in os.listdir(image_dir):
		folder_full = os.path.join(image_dir, folder)
		if os.path.isdir(folder_full):
			flag = False
			for file in os.listdir(folder_full):
				if file.endswith(ext):
					flag = True
					i += 1

			if flag == False:
				print(folder + " Does not contain image!")	
			else:
				c += 1		

	print("Images: ", i)	
	print("Classes: ", c)		
	return i, c		

def read_bbox_jason(image_path, image_dir):
	jason_file = os.path.splitext(image_path)[0] + '.json'
	jason_path = os.path.join(image_dir, jason_file)
	if os.path.isfile(jason_path):
		with open(jason_path) as json_data:
			d = json.load(json_data)
			# assume single bbox (Could work for logosc300)
			return d['detections_coordinates']
	else:	
		print("Error: "+jason_path+" DOES NOT exist.")
		return 0

# main
if __name__ == '__main__':
	
	train_list = os.path.join(dataset_path, 'split/train.txt')
	test_list = os.path.join(dataset_path, 'split/test.txt')
	patch_dir = os.path.join(dataset_path, 'patches')
	class_list = os.path.join(dataset_path, 'split/class_map.npy')

	pos_n = 5
	neg_n = 5
	# measured by pixels
	min_size = 40
	iou_th = 0.75

	start = time.time()
	
	# build or clean the patch directories
	if not os.path.exists(patch_dir): 
		os.makedirs(patch_dir)
	else:
		clear_dir(patch_dir)

	class_map = build_dir_structure(patch_dir, class_list)

	
	print("==> Generating patches for training set ...")
	# patch is indexed from 1 and from training images
	
	n = generate_patches(train_list, image_dir, patch_dir, class_map, pos_n, neg_n, min_size, iou_th, 'train', 1)
	
	
	print("==> Generating patches for test set ...")
	generate_patches(test_list, image_dir, patch_dir, class_map, pos_n, neg_n, min_size, iou_th, 'test', n)

	print("Time: "+str(datetime.timedelta(seconds=(time.time()-start))))
	
	



	