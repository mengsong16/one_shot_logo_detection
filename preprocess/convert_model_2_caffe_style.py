import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torch.optim
import math
import os
import scipy.misc 

name_map = {
	# =============== conv1 ================== #
	"features.0.weight": "conv1_1.weight",
	"features.0.bias": "conv1_1.bias",
	"features.1.weight": "bn1_1.weight",
	"features.1.bias": "bn1_1.bias",
	"features.1.running_mean": "bn1_1.running_mean",
	"features.1.running_var": "bn1_1.running_var",
	"features.3.weight": "conv1_2.weight",
	"features.3.bias": "conv1_2.bias",
	"features.4.weight": "bn1_2.weight",
	"features.4.bias": "bn1_2.bias",
	"features.4.running_mean": "bn1_2.running_mean",
	"features.4.running_var": "bn1_2.running_var",
	# =============== conv2 ================== #
	"features.7.weight": "conv2_1.weight",
	"features.7.bias": "conv2_1.bias",
	"features.8.weight": "bn2_1.weight",
	"features.8.bias": "bn2_1.bias",
	"features.8.running_mean": "bn2_1.running_mean",
	"features.8.running_var": "bn2_1.running_var",
	"features.10.weight": "conv2_2.weight",
	"features.10.bias": "conv2_2.bias",
	"features.11.weight": "bn2_2.weight",
	"features.11.bias": "bn2_2.bias",
	"features.11.running_mean": "bn2_2.running_mean",
	"features.11.running_var": "bn2_2.running_var",
	# =============== conv3 ================== #
	"features.14.weight": "conv3_1.weight",
	"features.14.bias": "conv3_1.bias",
	"features.15.weight": "bn3_1.weight",
	"features.15.bias": "bn3_1.bias",
	"features.15.running_mean": "bn3_1.running_mean",
	"features.15.running_var": "bn3_1.running_var",
	"features.17.weight": "conv3_2.weight",
	"features.17.bias": "conv3_2.bias",
	"features.18.weight": "bn3_2.weight",
	"features.18.bias": "bn3_2.bias",
	"features.18.running_mean": "bn3_2.running_mean",
	"features.18.running_var": "bn3_2.running_var",
	"features.20.weight": "conv3_3.weight",
	"features.20.bias": "conv3_3.bias",
	"features.21.weight": "bn3_3.weight",
	"features.21.bias": "bn3_3.bias",
	"features.21.running_mean": "bn3_3.running_mean",
	"features.21.running_var": "bn3_3.running_var",
	# =============== conv4 ================== #
	"features.24.weight": "conv4_1.weight",
	"features.24.bias": "conv4_1.bias",
	"features.25.weight": "bn4_1.weight",
	"features.25.bias": "bn4_1.bias",
	"features.25.running_mean": "bn4_1.running_mean",
	"features.25.running_var": "bn4_1.running_var",
	"features.27.weight": "conv4_2.weight",
	"features.27.bias": "conv4_2.bias",
	"features.28.weight": "bn4_2.weight",
	"features.28.bias": "bn4_2.bias",
	"features.28.running_mean": "bn4_2.running_mean",
	"features.28.running_var": "bn4_2.running_var",
	"features.30.weight": "conv4_3.weight",
	"features.30.bias": "conv4_3.bias",
	"features.31.weight": "bn4_3.weight",
	"features.31.bias": "bn4_3.bias",
	"features.31.running_mean": "bn4_3.running_mean",
	"features.31.running_var": "bn4_3.running_var",
	# =============== conv5 ================== # 
	"features.34.weight": "conv5_1.weight",
	"features.34.bias": "conv5_1.bias",
	"features.35.weight": "bn5_1.weight",
	"features.35.bias": "bn5_1.bias",
	"features.35.running_mean": "bn5_1.running_mean",
	"features.35.running_var": "bn5_1.running_var",
	"features.37.weight": "conv5_2.weight",
	"features.37.bias": "conv5_2.bias",
	"features.38.weight": "bn5_2.weight",
	"features.38.bias": "bn5_2.bias",
	"features.38.running_mean": "bn5_2.running_mean",
	"features.38.running_var": "bn5_2.running_var",
	"features.40.weight": "conv5_3.weight",
	"features.40.bias": "conv5_3.bias",
	"features.41.weight": "bn5_3.weight",
	"features.41.bias": "bn5_3.bias",
	"features.41.running_mean": "bn5_3.running_mean",
	"features.41.running_var": "bn5_3.running_var",
	# =============== classifier ================== # 
	"classifier.0.weight": "classifier.0.weight",
	"classifier.0.bias": "classifier.0.bias",
	"classifier.3.weight": "classifier.3.weight",
	"classifier.3.bias": "classifier.3.bias",
	"classifier.6.weight": "classifier.6.weight",
	"classifier.6.bias": "classifier.6.bias"

	}
	
if __name__ == '__main__':
	pretrain_path = "/data/uvn/pretrained/vgg16_bn.pth"
	new_pretrain_path = "/data/uvn/pretrained/vgg16_bn_caffe_style.pth"
	
	print('==> Converting nn model from pytorch style naming to caffe sytle naming.')
	pretrained_dict = torch.load(pretrain_path)
	
	new_pretrained_dict = {}
	for k, v in pretrained_dict.items():
		new_k = name_map[k]
		print(str(k)+" ==> "+ str(new_k)+": "+str(v.size()))	
		new_pretrained_dict[new_k] = v
	
	torch.save(new_pretrained_dict, new_pretrain_path)
	
	print("==> Checking correctness...")
	pretrained_dict_check = torch.load(new_pretrain_path)
	i = 0
	for k, v in pretrained_dict_check.items():
		print(str(k)+": "+str(v.size()))
		#if "conv" in k:
		#	print(v)
		i += 1

	print("==> Generated "+str(i)+" new layers.")	
	print('==> Done.')