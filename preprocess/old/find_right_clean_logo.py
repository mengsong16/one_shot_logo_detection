import argparse
import os
import shutil
import time
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision import transforms, utils
import torchvision.datasets as datasets
import torchvision.models as models
from feature_extraction import *
from PIL import Image
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchsample.transforms import RandomRotate, RandomBrightness, RandomAffine
import collections
import random
import json
import xmltodict

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

def crop_bbox_image(image_dir, image_path):
	image = Image.open(os.path.join(image_dir, image_path))
	image = image.convert('RGB')
	xmin, ymin, xmax, ymax = read_bbox_xml(image_path, image_dir)
	img_p = image.crop((xmin, ymin, xmax, ymax))

	return img_p

# read mean image
def read_mean_image(path):
	R = 0
	G = 0
	B = 0
	with open(path, 'r') as rf:
		line = rf.readline()
		R,G,B = line.strip().split()

	return float(R)/255.0, float(G)/255.0, float(B)/255.0	

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

# computer for accumulating, averaging and storing value
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

# assume jpg
def image_loader(path):
	ext = os.path.splitext(path)[1]
	with open(path, 'rb') as f:
		with Image.open(f) as image:
			return image.convert('RGB')
			#image_np = np.array(image)
			#image_np = image_np / 255.0
			
			#image = Image.fromarray(image_np)
			#return image_np

# read and transform one image
def read_trans_one_image(path, transformer):
	# read image
	img = image_loader(path)
	#print(list(img.getdata()))
	#print(img)
	# transform
	trans_img = transformer(img)

	return trans_img
	
def get_vector(image_path, transformer, model):
	# get input image

	img = read_trans_one_image(image_path, transformer)
	#plt.imsave(os.path.join(logo_vector_dir, os.path.join(folder, file)), source_img.numpy().transpose(1,2,0))
	
	img_var = torch.autograd.Variable(img)
	img_var = img_var.unsqueeze(0) # add one dimension
	# compute output vector
	vec = model(img_var)

	return vec

def get_vector_bbox(patch, transformer, model):
	#print(patch.size)
	#print(list(patch.getdata()))
	# get input image
	trans_patch = transformer(patch)
	#plt.imsave(os.path.join(logo_vector_dir, os.path.join(folder, file)), source_img.numpy().transpose(1,2,0))
	
	img_var = torch.autograd.Variable(trans_patch)
	img_var = img_var.unsqueeze(0) # add one dimension
	# compute output vector
	vec = model(img_var)

	return vec	

# extract clean logo feature vector and save
def find_right_clean_logo(model, transformer, logo_dir, image_dir, logo_mapping_file):
	
	# switch to evaluate mode (ff only)
	model.eval()

	# convert png to jpg
	brand_n = 0
	clean_logo_n = 0
	image_n = 0

	brand_time = AverageMeter()
	end = time.time()

	with open(logo_mapping_file, 'w') as f:
		#logo_lists = ["Adidas", "Pringles", "Apple", "Boston_Bruins"]
		#image_lists = ["glssubmittolive~7de301b1f0687885e10a8d6b4707ebe5.jpg"]
		for brand_name in os.listdir(logo_dir):
		#for brand_name in logo_lists:
			# for current brand
			if os.path.isdir(os.path.join(logo_dir, brand_name)):
				print(brand_name)
				logo_brand_folder = os.path.join(logo_dir, brand_name)
				image_brand_folder = os.path.join(image_dir, brand_name)
				#image_brand_folder = os.path.join(image_dir, "Adidas")
				
				# compute fc7 features for all clean logos 
				clean_logo_vecs = []
				for logo_image_name in os.listdir(logo_brand_folder):
					if logo_image_name.endswith(".jpg"):
						logo_image_path = os.path.join(logo_brand_folder, logo_image_name)
						logo_vec = get_vector(logo_image_path, transformer, model)
						clean_logo_vecs.append(logo_vec)
						clean_logo_n += 1

				#clean_logo_vecs = torch.cat(clean_logo_vecs, 0)	

				# compute fc7 features for all real images
				for image_name in os.listdir(image_brand_folder):
				#for image_name in image_lists:
					if image_name.endswith(".jpg"):
						image_path = os.path.join(image_brand_folder, image_name)
						bbox_patch = crop_bbox_image(image_dir, image_path)
						image_vec = get_vector_bbox(bbox_patch, transformer, model)

						# compare cosine distance
						index = 0
						best_index = -1
						max_sim = -10000
						for logo_vec in clean_logo_vecs:
							sim = torch.nn.functional.cosine_similarity(image_vec, logo_vec)
							sim_data = abs(sim.data[0])
							#print(str(index))
							#print(str(sim_data))
							if max_sim < sim_data:
								max_sim = sim_data
								best_index = index

							index += 1

						# record best index	
						f.write(os.path.join(brand_name, os.path.splitext(image_name)[0])+" "+str(best_index)+".jpg\n")

						# next turn
						image_n += 1

						#break

				# measure elapsed time
				brand_time.update(time.time() - end)
				end = time.time()

				# verbose
				if (brand_n+1) % 5 == 0:
					print('{0} brands Done\n'
					'brand avg time: {brand_time.avg:.3f} \n'.format(
					brand_n+1, brand_time=brand_time))
				
				brand_n += 1

				#break


	print("Processed " + str(brand_n) + " brands.")
	print("\t" + str(clean_logo_n) + " logos.")
	print("\t" + str(image_n) + " images.")
	


###  main
if __name__ == '__main__':

	###  hyper parameters
	use_cuda = torch.cuda.is_available()

	input_size = 224  # resize input image to square

	###  Paths
	root_dir = "/work/meng/data/logosc_300"
	logo_dir = os.path.join(root_dir, 'clean_logos')
	image_dir = os.path.join(root_dir, 'logosc300')
	
	logo_mapping_file = os.path.join(logo_dir, "image_logo_mapping.txt")
	
	# pretrained model path
	pretrained_dir = "/work/meng/uvn/pretrained/vgg16.pth"

	### redirect standard output
	# redirect output 
	#sys.stdout = open(os.path.join(logo_dir, 'fact_mapping.txt'), 'w')

	### data loading
	mean_R, mean_G, mean_B = read_mean_image(os.path.join(root_dir, "split/mean.txt"))

	print("Loaded mean: "+str([mean_R, mean_G, mean_B]))

	
	standard_transform = transforms.Compose(
		[transforms.Scale((input_size,input_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[mean_R, mean_G, mean_B], std=[1,1,1])
		])
	
	# create model
	print('==> Creating model...')	
	model = create_model(pretrain_path=pretrained_dir)
			

	###  set GPU and parallelization
	print('==> Found ' + str(torch.cuda.device_count()) + ' GPUs')
	if use_cuda:
		print("Using GPUs: cuda is available")
		model.cuda()
		gpu_usage = [0]
		model = torch.nn.DataParallel(model, device_ids=gpu_usage)
		print("Parallelizing " + str(gpu_usage) + " GPUs")
		# cudnn needs extra memory, turn it on while reducing the batch size. Otherwise, training begins without release test memory
		cudnn.benchmark = True
		if cudnn.benchmark == True:
			print("Using cudnn")
		else:
			print("NOT using cudnn")	
	else:
		print("NOT using GPUs: cuda is NOT available")	 


	# extract clean logo feature vector and save 
	find_right_clean_logo(model, standard_transform, logo_dir, image_dir, logo_mapping_file)

	

