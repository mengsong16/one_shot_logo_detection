import math
import random
import string
import unittest
import itertools
import contextlib
import warnings
import pickle
from copy import deepcopy
from itertools import repeat, product
from functools import wraps, reduce
from operator import mul

import numpy as np
from PIL import Image
from torchvision import transforms, utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as dp
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.legacy.nn as legacy
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.nn import Parameter


'''
TEST_CUDA = torch.cuda.is_available()
TEST_CUDNN = TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1))
TEST_CUDNN_VERSION = TEST_CUDNN and torch.backends.cudnn.version()
PRECISION = 1e-5
'''
class STN(nn.Module):
	def __init__(self):
		super(STN, self).__init__()
		print("Creating STN layer ...")

	# transform input image according to input theta	
	# input_theta: [N, 2, 3]
	# input_image: [N, C, H, W]
	# output_image: [N, C, H, W]
	def forward(self, input_image, input_theta):
		input_size = input_image.size()
		# checking shape
		theta_size = input_theta.size()
		assert theta_size[0] == input_size[0]
		assert theta_size[1] == 2
		assert theta_size[2] == 3
		# transform input theta to grid
		#print(input_size)
		grid = F.affine_grid(input_theta, input_size)
		#print(grid.size())

		# transform input image
		output_image = F.grid_sample(input_image, grid)
		# checking shape
		output_size = output_image.size()
		assert input_size == output_size

		return output_image

def theta2matrix(theta=0, tx=0, ty=0, scale=1):
	m = np.zeros((2,3))
	m[0,2] = tx
	m[1,2] = ty
	m[0,0] = math.cos(math.pi*theta/180) * scale
	m[1,1] = math.cos(math.pi*theta/180) * scale
	m[0,1] = math.sin(math.pi*theta/180) 
	m[1,0] = -math.sin(math.pi*theta/180) 

	return m
def weiredmatrix():
	m = np.zeros((2,3))
	m[0,2] = 100
	m[1,2] = 200
	m[0,0] = 500
	m[1,1] = -500
	m[0,1] = 300 
	m[1,0] = 400
	return m
	
###  main
if __name__ == '__main__':
	# create net
	net = STN()
	# create theta
	theta1 = theta2matrix(theta=30)
	theta2 = theta2matrix(scale=0.8)
	theta3 = theta2matrix(tx=100, ty=-100)
	theta4 = weiredmatrix()
	#print(theta1)
	theta_tensor = torch.from_numpy(theta4).float()
	theta_tensor = theta_tensor.unsqueeze_(0)
	theta_var = torch.autograd.Variable(theta_tensor)
	

	# read cat image
	with open('./cat.jpg', 'rb') as f:
		with Image.open(f) as img:
			img = img.convert('RGB')
			img_tensor = transforms.ToTensor()(img).float()
			img_tensor = img_tensor.unsqueeze_(0)
			img_var = torch.autograd.Variable(img_tensor)

			print(img_var.size())
			print(theta_var.size())
			out_var = net(img_var, theta_var)
			out_var = torch.squeeze(out_var, 0)
			print(out_var.size())
			out_pic = transforms.ToPILImage()(out_var.data)
			out_pic.save('./cat_trans.jpg')

