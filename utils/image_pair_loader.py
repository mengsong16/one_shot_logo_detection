import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchsample.transforms import RandomRotate, RandomBrightness, RandomAffine
import matplotlib.pyplot as plt
import collections
import random

from PIL import Image
import os.path
import scipy.misc
import sys


def read_mean_image(path):
	R = 0
	G = 0
	B = 0
	with open(path, 'r') as rf:
		line = rf.readline()
		R,G,B = line.strip().split()

	print("Mean image loaded.")	

	return float(R)/255.0, float(G)/255.0, float(B)/255.0	

# do not use io.imread(path), otherwise there will be error "'int' object is not subscriptable"
def default_image_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as image:
			# assume input image has 3 channels (not .png)
			# gray image will be converted to three chanel
			#img = np.array(img)/255.0 for imshow
			return image.convert('RGB')	
		
# format: patches/positives/_.jpg clean_logos/_.jpg label patch_class_id clean_logo_class_id
def default_flist_reader(flist):
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			patch_img_path, clean_img_path, label, _, _, = line.strip().split(',')
			imlist.append( (patch_img_path, clean_img_path, int(label)) )
					
	return imlist

def hm_flist_reader(flist):
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			source_img_path, target_img_path, label, src_cls_id, tgt_cls_id  = line.strip().split(',')
			imlist.append( (source_img_path, target_img_path, int(label), int(src_cls_id), int(tgt_cls_id)) )
					
	return imlist

def get_downsampled_images(source_imgs, target_imgs):
	N = source_imgs.size()[0]
	downsample_size = 64
	small_source_tensors = torch.zeros(source_imgs.size()[0], source_imgs.size()[1], downsample_size, downsample_size)
	small_target_tensors = torch.zeros(target_imgs.size()[0], target_imgs.size()[1], downsample_size, downsample_size)

	for j in list(range(N)):
		small_source_img = scipy.misc.imresize(source_imgs[j].numpy().transpose(1,2,0), (downsample_size,downsample_size))
		small_target_img = scipy.misc.imresize(target_imgs[j].numpy().transpose(1,2,0), (downsample_size,downsample_size))
		
		small_source_tensors[j] = transforms.ToTensor()(small_source_img)
		small_target_tensors[j] = transforms.ToTensor()(small_target_img)

	return 	small_source_tensors, small_target_tensors

def get_downsampled_images_source(source_imgs):
	N = source_imgs.size()[0]
	downsample_size = 64
	small_source_tensors = torch.zeros(source_imgs.size()[0], source_imgs.size()[1], downsample_size, downsample_size)
	
	for j in list(range(N)):
		small_source_img = scipy.misc.imresize(source_imgs[j].numpy().transpose(1,2,0), (downsample_size,downsample_size))
		small_source_tensors[j] = transforms.ToTensor()(small_source_img)
		

	return 	small_source_tensors

# load both positive and negative pairs from file	
# data format: source_image_path, target_image_path, label
class ImagePairDataset(Dataset):
	def __init__(self, root, flist, flist_reader=default_flist_reader, loader=default_image_loader, 
		source_transform=None, target_transform=None):
		self.root   = root	
		self.target_transform = target_transform
		self.source_transform = source_transform
		self.loader = loader

		self.imlist = flist_reader(flist)

		print("Read "+str(len(self.imlist))+" pairs.")


	def __getitem__(self, index):
		source_img_path, target_img_path, label = self.imlist[index]
		source_img = self.loader(os.path.join(self.root, source_img_path))
		target_img = self.loader(os.path.join(self.root, target_img_path))
		#print(os.path.join(self.root, source_img_path)+" "+os.path.join(self.root, target_img_path))
		
		if self.target_transform is not None:
			target_img = self.target_transform(target_img)
		if self.source_transform is not None:
			source_img = self.source_transform(source_img)	
		
		
		return source_img, target_img, label

	def __len__(self):
		return len(self.imlist)

# load both positive and negative pairs from file	
# data format: source_image_path, target_image_path, label, src_cls_id, tgt_cls_id
class ImagePairDatasetHM(Dataset):
	def __init__(self, root, flist, flist_reader=hm_flist_reader, loader=default_image_loader, 
		source_transform=None, target_transform=None):
		self.root   = root	
		self.target_transform = target_transform
		self.source_transform = source_transform
		self.loader = loader

		self.imlist = flist_reader(flist)	

		print("Read "+str(len(self.imlist))+" pairs.")


	def __getitem__(self, index):
		source_img_path, target_img_path, label, src_cls_id, tgt_cls_id = self.imlist[index]
		source_img = self.loader(os.path.join(self.root, source_img_path))
		target_img = self.loader(os.path.join(self.root, target_img_path))
		#print(os.path.join(self.root, source_img_path)+" "+os.path.join(self.root, target_img_path))
		
		if self.target_transform is not None:
			target_img = self.target_transform(target_img)
		if self.source_transform is not None:
			source_img = self.source_transform(source_img)	
		
		
		return source_img, target_img, label, src_cls_id, tgt_cls_id

	def __len__(self):
		return len(self.imlist)

	def get_imlist(self):
		return self.imlist			

# dynamically load negative pairs, positive paris are loaded from file, without class id
class ImagePairDatasetDynamic(Dataset):
	def __init__(self, root, pos_flist, neg_list, flist_reader=default_flist_reader, loader=default_image_loader, 
		source_transform=None, target_transform=None):
		self.root   = root	
		self.target_transform = target_transform
		self.source_transform = source_transform
		self.loader = loader
		#print(neg_list)
		pos_list = flist_reader(pos_flist)
		print("Read "+str(len(pos_list))+" positive pairs.")
		self.imlist = pos_list + neg_list
		print("Get "+str(len(self.imlist))+" pairs in total.")
		
		

	def __getitem__(self, index):
		source_img_path, target_img_path, label = self.imlist[index]
		source_img = self.loader(os.path.join(self.root, source_img_path))
		target_img = self.loader(os.path.join(self.root, target_img_path))
		#print(os.path.join(self.root, source_img_path)+" "+os.path.join(self.root, target_img_path))
		
		if self.target_transform is not None:
			target_img = self.target_transform(target_img)
		if self.source_transform is not None:
			source_img = self.source_transform(source_img)	
		
		
		return source_img, target_img, label

	def __len__(self):
		return len(self.imlist)

# dynamically load negative pairs, positive paris are loaded from file, with class id
class ImagePairDatasetHM_Neg(Dataset):
	def __init__(self, root, pos_flist, neg_list, flist_reader=hm_flist_reader, loader=default_image_loader, 
		source_transform=None, target_transform=None):
		self.root   = root	
		self.target_transform = target_transform
		self.source_transform = source_transform
		self.loader = loader

		pos_list = flist_reader(pos_flist)
		print("Read "+str(len(pos_list))+" positive pairs.")
		self.imlist = pos_list + neg_list
		print("Get "+str(len(self.imlist))+" pairs in total.")
			


	def __getitem__(self, index):
		source_img_path, target_img_path, label, src_cls_id, tgt_cls_id = self.imlist[index]
		source_img = self.loader(os.path.join(self.root, source_img_path))
		target_img = self.loader(os.path.join(self.root, target_img_path))
		#print(os.path.join(self.root, source_img_path)+" "+os.path.join(self.root, target_img_path))
		
		if self.target_transform is not None:
			target_img = self.target_transform(target_img)
		if self.source_transform is not None:
			source_img = self.source_transform(source_img)	
		
		
		return source_img, target_img, label, src_cls_id, tgt_cls_id

	def __len__(self):
		return len(self.imlist)

	def get_imlist(self):
		return self.imlist

# both positive and negative pairs are generated
class ImagePairDatasetDynamicHM(Dataset):
	def __init__(self, root, pos_list, neg_list, loader=default_image_loader, source_transform=None, target_transform=None):
		self.root = root	
		self.target_transform = target_transform
		self.source_transform = source_transform
		self.loader = loader
		
		self.imlist = pos_list + neg_list
		print("Generate "+str(len(self.imlist))+" pairs in total.")
		
		

	def __getitem__(self, index):
		source_img_path, target_img_path, label, src_cls_id, tgt_cls_id = self.imlist[index]
		source_img = self.loader(os.path.join(self.root, source_img_path))
		target_img = self.loader(os.path.join(self.root, target_img_path))
		#print(os.path.join(self.root, source_img_path)+" "+os.path.join(self.root, target_img_path))
		
		if self.target_transform is not None:
			target_img = self.target_transform(target_img)
		if self.source_transform is not None:
			source_img = self.source_transform(source_img)	
		
		
		return source_img, target_img, label, src_cls_id, tgt_cls_id

	def __len__(self):
		return len(self.imlist)		

# My implementation of Scale
# size = (W,H)
class MyScale():
	def __init__(self, size, interpolation=Image.BILINEAR):
		assert isinstance(size, collections.Iterable) and len(size) == 2
		self.size = size
		self.interpolation = interpolation

	def __call__(self, img):
		"""
		Args:
			img (PIL.Image): Image to be scaled.
		Returns:
			PIL.Image: Rescaled image.
		"""
		return img.resize(self.size, self.interpolation)


# main
if __name__ == '__main__':
	root_dir = "/work/meng"
	my_batch_size = 2
	'''
	# normalize: Given mean: (R, G, B) and std: (R, G, B), should be performed after ToTensor
	# totensor: from H*W*C to C*H*W
	dataset_path = os.path.join(root_dir, "data/logosc_300")
	mean_R, mean_G, mean_B = read_mean_image(os.path.join(dataset_path, "split/mean.txt"))

	# RandomSizedCrop(227), RandomHorizontalFlip()
	# torchsample.transforms.AffineCompose
	# transforms.Scale(227),transforms.CenterCrop(227)

	# transforms.Scale: input PIL.Imag
	# transforms.Normalize: input [0,1]
	# RandomAffine and other affine transformation from torchsample need image valued in [0,1]

	standard_transform = transforms.Compose(
			[transforms.Scale((224,224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[mean_R, mean_G, mean_B], std=[1,1,1])
			])
	
	
	if random.random() < 0.5:
		train_transform_instance = transforms.Compose(
			[transforms.Scale((224,224)),
			transforms.ToTensor(),
			RandomAffine(rotation_range=90, shear_range=30, zoom_range=(0.8,1.2)),
			transforms.Normalize(mean=[mean_R, mean_G, mean_B], std=[1,1,1])
			])
	else:
		train_transform_instance = standard_transform


	test_transform_instance = standard_transform


	train_dataset_instance = ImagePairDataset(root=dataset_path, 
		flist=os.path.join(dataset_path, "pairs/train.txt"),
		source_transform=train_transform_instance,
		target_transform=standard_transform)

	test_dataset_instance = ImagePairDataset(root=dataset_path, 
		flist=os.path.join(dataset_path, "pairs/test.txt"),
		source_transform=standard_transform)
	# shuffle: have the data reshuffled at every epoch 
	# pin_memory: accelarate the speed of accessing CPU memory

	train_loader = torch.utils.data.DataLoader(
		train_dataset_instance,
		batch_size=my_batch_size, shuffle=True,
		num_workers=8, pin_memory=True)

	test_loader = torch.utils.data.DataLoader(
		test_dataset_instance,
		batch_size=my_batch_size, shuffle=False,
		num_workers=8, pin_memory=True)
	
	# Iterate through the loader, one batch per iteration
	i = 0
	for source_imgs, target_imgs, labels in train_loader:
		if i >= 1:
			break
		
		
		for j in list(range(my_batch_size)):
			# numpy().transpose(1,2,0)
			# transforms.ToPILImage(source_imgs[j])
			scipy.misc.imsave(os.path.join(dataset_path, "tmp/source/"+str(j)+".jpg"), source_imgs[j].numpy().transpose(1,2,0))
			scipy.misc.imsave(os.path.join(dataset_path, "tmp/target/"+str(j)+".jpg"), target_imgs[j].numpy().transpose(1,2,0))
			#print(source_imgs[j].numpy().transpose(1,2,0))
			# in (0,1)
			small_source_img = scipy.misc.imresize(source_imgs[j].numpy().transpose(1,2,0), (64,64))
			small_target_img = scipy.misc.imresize(target_imgs[j].numpy().transpose(1,2,0), (64,64))
			# in (0, 255)
			#print(small_source_img)
			small_source_tensor = transforms.ToTensor()(small_source_img)
			print(small_source_img.shape)
			#print(small_source_tensor)
			print(small_source_tensor.size())
			scipy.misc.imsave(os.path.join(dataset_path, "tmp/source/"+str(j)+"_small.jpg"), small_source_img)
			scipy.misc.imsave(os.path.join(dataset_path, "tmp/target/"+str(j)+"_small.jpg"), small_target_img)
			#print(labels[j])
		
		i += 1
	'''
	print("Done.")	
