import sys

import argparse
import os
import shutil
import time
import sys
import datetime
# since display is undeclared
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.image_pair_loader import *
from models.pair_vgg import *
from models.compute_embedding_matrix import *
from config import *

class Logger(object):
	def __init__(self, f1, f2):
		self.f1, self.f2 = f1, f2

	def write(self, msg):
		self.f1.write(msg)
		self.f2.write(msg)
		
	def flush(self):
		pass

# load matched model, not consider optimizer
def load_matched_model_only(model, pretrain_path):
	pretrained_dict = torch.load(pretrain_path)
	model_dict = model.state_dict()
	
	i = 0
	for param in model.parameters():
		i += 1
	print("model size: "+str(i))
	print("model_dict size: "+str(len(model_dict)))	
	
	
	# 1. filter out unnecessary keys
	#prev_len = len(pretrained_dict)	
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size()==v.size()}
	cur_len = len(pretrained_dict)	
	
	# 2. overwrite entries in the existing state dict
	# if some key in pretrained_dict do no exist in model_dict, error will be raised
	# matching is done by name	
	model_dict.update(pretrained_dict)
	
	# 3. load the new state dict
	# update all parameters and buffers in the state_dict of model
	model.load_state_dict(model_dict)

	# print loaded and not loaded layers
	# old_layer_names may contain the layers in model_dict, but not in model
	print('[model_dict] Matched layers: '+ str(cur_len)+' / '+str(len(model_dict)))
	old_layer_names = []
	for k, v in pretrained_dict.items():
		print("	"+str(k)+": "+str(v.size()))
		old_layer_names.append(k)

	print('[model_dict] Not matched layers: '+ str(len(model_dict)-cur_len)+' / '+str(len(model_dict)))
	for k, v in model_dict.items():
		if k not in pretrained_dict:
			print("	"+str(k)+": "+str(v.size()))

	return model

# create model, not consider optimizer
def create_model(model_name="", pretrained=False, pretrain_path=""):
	print('==> Creating model from scratch...')
	print("Model name: "+model_name)
	# create model
	class_num = 2
	model = create_model_sub(model_name, class_num)
	
	if pretrained:
		print('==> Creating model from pretrained model: '+pretrain_path)
		print('==> Loading pretrained model...')
		model = load_matched_model_only(model, pretrain_path)
			
	return model

# resume a model from checkpoint
# resume epoch is indexed from 0
def resume_model_checkpoint(checkpoint_path, resume_epoch, model):
	resume_path = os.path.join(checkpoint_path, "checkpoint_" + str(resume_epoch) + ".pth")
	if os.path.isfile(resume_path):
		print("=> Resuming from checkpoint...")
		print("=> Loading checkpoint '{}'".format(resume_path))
		# load checkpoint
		checkpoint = torch.load(resume_path)
		# load model
		model.load_state_dict(checkpoint['state_dict'])
		loss_avg = checkpoint['loss_avg']
		
		print("=> Loaded checkpoint '{}' (epoch {})".format(resume_path, resume_epoch))
		print("Avg loss of resummed epoch: %.3f" %(loss_avg))
	else:
		print("=> No checkpoint found at '{}'".format(resume_path))

	
	return model, loss_avg

# test
def validate(model, pos_patch_file, neg_patch_file, clean_logo_file, cleanlogo_path, patch_path, cur_output_path, standard_transform, k_vec):
	# switch to evaluate mode
	model.eval()
	clean_logo_class_ids, pos_patch_class_ids = compute_pos_neg_matrices(standard_transform, model, pos_patch_file, neg_patch_file, clean_logo_file, cleanlogo_path, patch_path, cur_output_path)
	compute_recall(cur_output_path, clean_logo_class_ids, pos_patch_class_ids, k_vec)



###  main
if __name__ == '__main__':
	
	###  arguments
	parser = argparse.ArgumentParser(description='PyTorch Pair VGG Testing')
	parser.add_argument('--finetune', action='store_true', help='finetune the model based on Imagenet')
	parser.add_argument('--freeze', action='store_true', help='freeze the pretrained layers')
	parser.add_argument('--end2end', action='store_true', help='finetune the model using results from previous step')
	parser.add_argument('--model', default='vgg16_stack', type=str, help='name of model architecture')
	parser.add_argument('--dynamic', action='store_true', help='dynamically generate negative pairs')
	parser.add_argument('--hard_mining', action='store_true', help='mininig hard pairs for each epoch')
	parser.add_argument('--negative_only', action='store_true', help='mininig only hard negative pairs for each epoch.')
	parser.add_argument('--unbalanced', action='store_true', help='during hard mining, do not keep balance between positives and negatives.')
	parser.add_argument('--start_epoch', default=0, type=int, help='index of epoch where we start testing (from 0)')
	parser.add_argument('--end_epoch', default=2, type=int, help='index of epoch where we end testing (from 0)')
	parser.add_argument('--opt', default='test_w32', type=str, help='name of test set we want to use')
	

	args = parser.parse_args()
	
	###  hyper parameters
	use_cuda = torch.cuda.is_available()
	val_step = 1 # For how many epoches should we do validation
	input_size = 224  # resize input image to square
	print_freq = 20
	load_num_workers = 32
	# start test from epoch x until y (indexed from 0)
	start_epoch = args.start_epoch
	end_epoch = args.end_epoch

	k_vec = [1, 3, 5, 10, 20, 40, 60, 80, 100]
	###  Paths
	# model data path
	if args.end2end:
		model_data_dir = os.path.join(work_dir, "exps_end2end")
	else:	
		model_data_dir = os.path.join(work_dir, "exps")

	if not os.path.exists(model_data_dir):
		os.makedirs(model_data_dir)

	# model path
	model_path = os.path.join(model_data_dir, args.model)
	if args.dynamic:
		model_path += "_dynamic"
		print("Use dynamic negative pair sampling")
	if args.hard_mining:
		model_path += "_hm"
		print("Use hard mining")	
		if args.negative_only:
			model_path += "_neg"
			print("Hard mining for negatives only.")
		if args.unbalanced:
			model_path += "_unbalanced"
			print("Hard mining with unbalanced positives and negatives.")	
		
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	# pretrained model path
	pretrained_dir = os.path.join(work_dir, "pretrained")
	if "_bn" in args.model: # load pretrained model from this path
		if "_deform" not in args.model:
			pretrain_path = os.path.join(pretrained_dir, "vgg16_bn.pth")
		else:
			pretrain_path = os.path.join(pretrained_dir, "vgg16_bn_caffe_style.pth")
	else:	
		pretrain_path = os.path.join(pretrained_dir, "vgg16.pth")
	if not os.path.exists(pretrain_path):
		print("Error: Pretrained model does not exist!")
	
	# experimental setting	
	# when testing, args.freeze is only used to pick the right directory
	# when training, args.freeze is also used to pick optimizing strategy
	if args.finetune:
		if args.freeze:
			exp_path = os.path.join(model_path, "finetune_freeze")
		else:
			exp_path = os.path.join(model_path, "finetune_all")
	else:
		exp_path = os.path.join(model_path, "scratch")
	if not os.path.exists(exp_path):
		os.makedirs(exp_path)
	# checkpoint path
	checkpoint_path = os.path.join(exp_path, "checkpoints") # resume model from this path
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	
	# test path
	# args.opt: test_w32, test_wo32, test_seen, val_unseen, val_seen
	test_path = os.path.join(exp_path, str(args.opt))
	if not os.path.exists(test_path):
		os.makedirs(test_path)

	if not os.path.exists(test_path):
		os.makedirs(test_path)

	# log path
	log_path = os.path.join(exp_path, "logs") 
	if not os.path.exists(log_path):
		os.makedirs(log_path)

	
	### redirect standard output
	### name log file as current time
	current_time = time.time()
	st = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
	logfile = open(os.path.join(log_path, str(args.opt)+'_'+st+'.txt'), 'w')
	sys.stdout = Logger(sys.stdout, logfile)

	# test file path
	if args.opt == "test_w32":
		pos_patch_file = os.path.join(test_patch_dir, "test_w32_pos.txt")
		neg_patch_file = os.path.join(test_patch_dir, "test_w32_neg.txt")
		clean_logo_file = os.path.join(csv_cleanlogo_dir, "test_w32_cleanlogos.txt")
		cleanlogo_path = test_clean_logo_dir
		patch_path = test_patch_dir
	elif args.opt == "test_wo32":
		pos_patch_file = os.path.join(test_patch_dir, "test_wo32_pos.txt")
		neg_patch_file = os.path.join(test_patch_dir, "test_wo32_neg.txt")
		clean_logo_file = os.path.join(csv_cleanlogo_dir, "test_wo32_cleanlogos.txt")
		cleanlogo_path = test_clean_logo_dir
		patch_path = test_patch_dir
	elif args.opt == "test_seen":
		pos_patch_file = os.path.join(test_patch_dir, "test_seen_pos.txt")
		neg_patch_file = os.path.join(test_patch_dir, "test_seen_neg.txt")
		clean_logo_file = os.path.join(csv_cleanlogo_dir, "test_seen_cleanlogos.txt")
		cleanlogo_path = test_clean_logo_dir
		patch_path = test_patch_dir
	elif args.opt == "val_unseen":
		pos_patch_file = os.path.join(val_patch_dir, "val_pos.txt")
		neg_patch_file = os.path.join(val_patch_dir, "val_neg.txt")
		clean_logo_file = os.path.join(csv_cleanlogo_dir, "val_cleanlogos.txt")
		cleanlogo_path = val_clean_logo_dir
		patch_path = val_patch_dir
	elif args.opt == "val_seen":
		pos_patch_file = os.path.join(val_patch_dir, "val_seen_pos.txt")
		neg_patch_file = os.path.join(val_patch_dir, "val_seen_neg.txt")
		clean_logo_file = os.path.join(csv_cleanlogo_dir, "val_seen_cleanlogos.txt")
		cleanlogo_path = val_clean_logo_dir
		patch_path = val_patch_dir
	else:
		print("Error: test option is incorrect")	

	# print out test options
	print('==> Test on set '+str(args.opt)+' ...')
	### print out arguments
	print('==> Basic settings...')
	print('start epoch: '+str(start_epoch))
	print('end epoch: '+str(end_epoch))

	if "_bn" in args.model :
		print('Use batch normalization')
	else:
		print('DO NOT use batch normalization')	

	### data loading
	mean_R, mean_G, mean_B = read_mean_image(os.path.join(train_dir, "mean.txt"))

	print('==> Preparing data...')

	# standard transformation
	standard_transform = transforms.Compose(
		[transforms.Scale((input_size,input_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[mean_R, mean_G, mean_B], std=[1,1,1])
		])

	
	### Create a model
	model = create_model(model_name=args.model, pretrained=args.finetune, pretrain_path=pretrain_path)
			
	
	###  set GPU and parallelization
	print('==> Found ' + str(torch.cuda.device_count()) + ' GPUs')
	if use_cuda:
		print("Using GPUs: cuda is available")
		model.cuda()
		gpu_usage = list(range(torch.cuda.device_count()))
		assert len(gpu_usage) == 1
		# model name has been changed after data parallel!
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

	
	###  testing
	# If pretrained and test from beginning, test pretrained model
	if args.finetune == True and start_epoch == 0:
	#if False:
		epoch = -1
		cur_time = time.time()
		cur_st = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d_%H:%M:%S')
		print('==> Test on pretrained model, epoch '+str(epoch)+': '+cur_st)

		cur_test_path = os.path.join(test_path, "pretrained")
		if not os.path.exists(cur_test_path):
			os.makedirs(cur_test_path)

		validate(model, pos_patch_file, neg_patch_file, clean_logo_file, cleanlogo_path, patch_path, cur_test_path, standard_transform, k_vec)
		
		# End time
		print('==> Pretrained model test completed: '+str(datetime.timedelta(seconds=time.time()-cur_time)))
	

	# test trained model
	# epoch is indexed from 0
	print('==> Start testing from epoch ' + str(start_epoch) + '...')
	start_time = time.time()
	start_st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M:%S')
	print('==> Start time: '+start_st)

	i = 0
	for epoch in list(range(start_epoch, end_epoch+1)):
		# always test the first and the last epoch, then test every val_step
		if (i == 0) or (i == end_epoch-start_epoch) or ((i + 1) % val_step == 0):
			# Start Time
			cur_time = time.time()
			cur_st = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d_%H:%M:%S')
			print('==> Epoch '+str(epoch)+': '+cur_st)

			
			# resume model
			model, _ = resume_model_checkpoint(checkpoint_path, epoch, model)
			# evaluate on test set
			cur_test_path = os.path.join(test_path, "epoch_"+str(epoch))
			if not os.path.exists(cur_test_path):
				os.makedirs(cur_test_path)

			#print(cur_test_path)	
			validate(model, pos_patch_file, neg_patch_file, clean_logo_file, cleanlogo_path, patch_path, cur_test_path, standard_transform, k_vec)
			# End time
			print('==> Epoch '+str(epoch)+': '+str(datetime.timedelta(seconds=time.time()-cur_time)))
		
		# next epoch
		i += 1
	
	# print summary
	end_time = time.time()
	end_st = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d_%H:%M:%S')
	print('==> End time: '+end_st)
	print('==> Total test time: '+str(datetime.timedelta(seconds=end_time-start_time)))
	print('Done.')