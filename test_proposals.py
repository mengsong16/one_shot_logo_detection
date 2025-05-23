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


class Logger(object):
	def __init__(self, f1, f2):
		self.f1, self.f2 = f1, f2

	def write(self, msg):
		self.f1.write(msg)
		self.f2.write(msg)
		
	def flush(self):
		pass

# find the largest rectangle bounding box
def get_rectangle(coords):
	x1,y1,x2,y2,x3,y3,x4,y4 = coords.strip().split(',')
	#print("%s, %s, %s, %s, %s, %s, %s, %s" %(x1,y1,x2,y2,x3,y3,x4,y4))
	x1 = int(x1)
	x2 = int(x2)
	x3 = int(x3)
	x4 = int(x4)
	y1 = int(y1)
	y2 = int(y2)
	y3 = int(y3)
	y4 = int(y4)
	left = min([x1,x2,x3,x4])
	right = max([x1,x2,x3,x4])
	upper = min([y1,y2,y3,y4])
	bottom = max([y1,y2,y3,y4])

	return left,upper,right,bottom

def read_files(proposal_dir, retrieval_dir, image_dir, verify_dir):
	images = []
	coord_files = []
	retrieval_files = []
	verify_file_paths = []

	for filename in os.listdir(proposal_dir):
		if filename.endswith('.txt'):
			coord_path = os.path.join(proposal_dir, filename)
			img_path = os.path.join(image_dir, os.path.splitext(filename)[0]+'.jpg')
			retrieval_path = os.path.join(retrieval_dir, filename)
			verify_path = os.path.join(verify_dir, filename)

			if not os.path.exists(img_path):
				print("Error: "+img_path+" Does NOT exist!")
				break
			if not os.path.exists(coord_path):
				print("Error: "+coord_path+" Does NOT exist!")
				break
			if not os.path.exists(retrieval_path):
				print("Error: "+retrieval_path+" Does NOT exist!")
				break

			images.append(img_path)
			coord_files.append(coord_path)
			retrieval_files.append(retrieval_path)
			verify_file_paths.append(verify_path)

	return images, coord_files, retrieval_files, verify_file_paths

# (H x W x 3) in range [0,255]
def default_image_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as image:
			return image.convert('RGB')	

def patch_loader(image_path, coord_file_path):

	img = default_image_loader(image_path)
	patches = []
	with open(coord_file_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			left, upper, right, bottom = get_rectangle(line)
			patches.append(img.crop((left, upper, right, bottom)))

	return patches		


def cleanlogo_loader(retrieval_file_path):
	clean_logo_matrix = []

	with open(retrieval_file_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			clean_logo_row = []
			paths = line.strip().split(',')
			for path in paths:
				clean_logo_row.append(default_image_loader(path))

			clean_logo_matrix.append(clean_logo_row)

	return clean_logo_matrix		



class ProposalDataset(Dataset):
	def __init__(self, proposal_dir, image_dir, retrieval_dir, verify_dir, file_reader=read_files, cleanlogo_loader=cleanlogo_loader, 
		patch_loader=patch_loader, transform=None):

		self.transform = transform
		self.cleanlogo_loader = cleanlogo_loader
		self.patch_loader = patch_loader

		self.images, self.coord_files, self.retrieval_files, self.verify_file_paths = file_reader(proposal_dir, retrieval_dir, image_dir, verify_dir)


		print(str(len(self.images)) + " images needs to be processed.")


	def __getitem__(self, index):
		image_path = self.images[index]
		coord_file_path = self.coord_files[index]
		retrieval_file_path = self.retrieval_files[index]

		patches = self.patch_loader(image_path, coord_file_path)
		clean_logo_matrix = self.cleanlogo_loader(retrieval_file_path)

		# organize into pair
		pair_matrix = []

		for i in list(range(len(patches))):
			pair_row = []
			patch = patches[i]
			clean_logo_row = clean_logo_matrix[i]

			for clean_logo in clean_logo_row:
				if self.transform is not None:
					pair_row.append((self.transform(patch), self.transform(clean_logo)))
				else:	
					pair_row.append((patch, clean_logo))

			pair_matrix.append(pair_row)
		
		
		#print(self.verify_file_paths[index])

		return pair_matrix, self.verify_file_paths[index]

	def __len__(self):
		return len(self.images)

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

# compute accuracy with predicted labels
def accuracy(output, target):
	"""Computes the precision@k for the specified values of k"""
	batch_size = target.size(0)
	# pred: B*1
	_, pred = output.topk(1, 1, True, True)
	# pred: 1*B
	pred = pred.t()
	# correct: 1*B
	correct = pred.eq(target.view(1, -1))
	# correct_1: scalar
	correct_1 = correct.view(-1).float().sum(0)
	res = correct_1.mul_(100.0 / batch_size)
	return res, pred


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
		acc_avg = checkpoint['accuracy_avg']
		print("=> Loaded checkpoint '{}' (epoch {})".format(resume_path, resume_epoch))
		print("Avg loss of resummed epoch: %.3f" %(loss_avg))
		print("Avg training accuracy of resummed epoch: %.3f%%" %(acc_avg))
	else:
		print("=> No checkpoint found at '{}'".format(resume_path))

	
	return model, loss_avg, acc_avg

# test proposal
def validate(val_loader, model, print_freq, model_name):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	pos_top1 = AverageMeter()
	neg_top1 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (pair_matrix, verify_file_path) in enumerate(val_loader):
		
		with open(verify_file_path[0], 'w') as f:
			for pair_row in pair_matrix:
				prob_row = []
				for (source_img, target_img) in pair_row:

					source_var = torch.autograd.Variable(source_img).cuda()
					target_var = torch.autograd.Variable(target_img).cuda()

					# compute output
					if "_stn" in model_name:
						if "_double" in model_name:
							small_source_img, small_target_img = get_downsampled_images(source_img, target_img)
							small_source_var = torch.autograd.Variable(small_source_img)
							small_target_var = torch.autograd.Variable(small_target_img)
							output = model(source_var, target_var, small_source_var, small_target_var)
						elif "_single" in model_name:
							small_source_img = get_downsampled_images_source(source_img)
							small_source_var = torch.autograd.Variable(small_source_img)
							output = model(source_var, target_var, small_source_var)
						elif "_late" in model_name:
							output = model(source_var, target_var)		
					else:
						output = model(source_var, target_var)

					outprob = torch.nn.functional.softmax(output)
					#print(outprob.data.cpu().numpy())
					#print(outprob.data.cpu().numpy()[0][1])
					prob_row.append(outprob.data.cpu().numpy()[0][1])
					
				# dump row to file
				for e in prob_row[:-1]:
					f.write(str(e)+' ')

				f.write(str(prob_row[-1])+'\n')
				
		# verbose
		if (i+1) % print_freq == 0:
			print("%d images completed."%(i+1))


###  main
if __name__ == '__main__':
	
	###  arguments
	parser = argparse.ArgumentParser(description='PyTorch Pair VGG Testing')
	parser.add_argument('--finetune', action='store_true', help='finetune the model based on Imagenet')
	parser.add_argument('--freeze', action='store_true', help='freeze the pretrained layers')
	parser.add_argument('--model', default='vgg16_stack', type=str, help='name of model architecture')
	parser.add_argument('--dynamic', action='store_true', help='dynamically generate negative pairs')
	parser.add_argument('--hard_mining', action='store_true', help='mininig hard pairs for each epoch')
	parser.add_argument('--negative_only', action='store_true', help='mininig only hard negative pairs for each epoch.')
	parser.add_argument('--unbalanced', action='store_true', help='during hard mining, do not keep balance between positives and negatives.')
	parser.add_argument('--opt', default='test_w32', type=str, help='name of test set we want to use')
	parser.add_argument('--epoch', default=10, type=int, help='index of the epoch we want to test on')
	parser.add_argument('--batch_size', default=1, type=int, help='batch size')
	
	args = parser.parse_args()
	
	###  hyper parameters
	use_cuda = torch.cuda.is_available()
	input_size = 224  # resize input image to square
	print_freq = 100
	load_num_workers = 0
	k = 10

	epoch = args.epoch
	spec = args.opt
	my_batch_size = args.batch_size

	###  Paths
	# model data path
	model_data_dir = os.path.join(work_dir, "exps")

	# model path (w or w/o dynamic sampling, hard mininig)
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

	# checkpoint path
	checkpoint_path = os.path.join(exp_path, "checkpoints") # resume models from this path
		
	# end2end
	end2end_dir = "/data/end2end"
	# retrieval path
	feature_embedding_dir = os.path.join(end2end_dir, "feature_embeddings")
	retrieval_dir = os.path.join(feature_embedding_dir, spec)
	# region proposal coordinates path
	region_proposal_dir = os.path.join(end2end_dir, "region_proposals")
	proposal_dir = os.path.join(os.path.join(region_proposal_dir, spec), "coord_pred_box")

	# output dir 
	verification_dir = os.path.join(end2end_dir, "verification")
	if not os.path.exists(verification_dir): 
		os.makedirs(verification_dir)
	
	outdir = os.path.join(verification_dir, spec)
	if not os.path.exists(outdir): 
		os.makedirs(outdir)
	else:
		clear_dir(outdir)
	
	# log dir 
	log_dir = os.path.join(end2end_dir, "logs")

	# image dir
	image_dir = test_image_dir
	
	### redirect standard output
	start_time = time.time()
	start_st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M:%S')
	logfile = open(os.path.join(log_dir, 'verification')+'_'+spec+'_'+start_st+'.txt', 'w')
	sys.stdout = Logger(sys.stdout, logfile)

	
	# print out test options
	print('==> Test on proposals')
	print('==> Test on set '+str(args.opt)+' ...')
	### print out arguments
	print('==> Basic settings...')
	print('Test epoch: '+str(epoch))

	if "_bn" in args.model:
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

	# dataset
	test_dataset_instance = ProposalDataset(proposal_dir, image_dir, retrieval_dir, outdir, transform=standard_transform)

	# data loader
	test_loader = torch.utils.data.DataLoader(
		test_dataset_instance,
		batch_size=my_batch_size, shuffle=False,
		num_workers=load_num_workers, pin_memory=True)

	### Create a model
	model = create_model(model_name=args.model, pretrained=False, pretrain_path="")
			
	### define loss function (criterion)
	if "_focal" in args.model:
		criterion = FocalLoss()
		print("Using focal loss")
	else:		
		criterion = nn.CrossEntropyLoss()
		print("Using cross entropy loss")

	###  set GPU and parallelization
	print('==> Found ' + str(torch.cuda.device_count()) + ' GPUs')
	if use_cuda:
		print("Using GPUs: cuda is available")
		model.cuda()
		gpu_usage = list(range(torch.cuda.device_count()))
		# model has been changed after data parallel!
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

	# test trained model
	# epoch is indexed from 0
	start_time = time.time()
	start_st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M:%S')
	print('==> Start time: '+start_st)

	# resume model
	model, _, _ = resume_model_checkpoint(checkpoint_path, epoch, model)
	# evaluate on test set
	# return avg accuracy of this epoch	
	validate(test_loader, model, print_freq, args.model)
	
	# print summary
	end_time = time.time()
	end_st = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d_%H:%M:%S')
	print('==> End time: '+end_st)
	print('==> Total test time: '+str(datetime.timedelta(seconds=end_time-start_time)))
	print('Done.')

		
