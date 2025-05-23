import sys

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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import scipy.misc

from utils.image_pair_loader import *
from models.pair_vgg import *
#from preprocess.single2pair_dynamic import *
#from preprocess.single2pair_hardmining import *
from config import *


class Logger(object):
		def __init__(self, f1, f2):
			self.f1, self.f2 = f1, f2

		def write(self, msg):
			self.f1.write(msg)
			self.f2.write(msg)
			
		def flush(self):
			pass

# prepare for dynamic negative sampling
def prepare_dynamic_negative_sampling(root_dir, clean_logo_png_path, patch_dir, class_list, gt, logo_rate = 0.5):
	# get clean logo map 
	clean_logo_map = read_clean_logo_map(os.path.join(os.path.join(root_dir, clean_logo_png_path), 'clean_logo_map.npy'))
	# logo bins tell us for each class, what images do we have, should be modified dynamically
	if gt == False:
		train_pos_patch_file = 'train_pos_list.txt'
	else:
		train_pos_patch_file = 'train_pos_gt_list.txt'

	logo_bins = build_logo_bins(os.path.join(os.path.join(root_dir, patch_dir), train_pos_patch_file))
	fixed_logo_bins = build_logo_bins(os.path.join(os.path.join(root_dir, patch_dir), train_pos_patch_file))
	# logo and no logo counts tell us for each class, how many logo negative pairs and no logo negative pairs do we need, should keep unchanged 
	logo_counts, no_logo_counts = split_logo_bins(fixed_logo_bins, logo_rate)
	# list of no logo patches
	no_logo_list = read_no_logo_patches(os.path.join(os.path.join(root_dir, patch_dir), 'train_neg_list.txt'))
	# map of class id and class name
	class_map = read_class_map_pos(class_list)

	return clean_logo_map, logo_bins, fixed_logo_bins, logo_counts, no_logo_counts, no_logo_list, class_map

# prepare for hard mining sampling
def prepare_hard_mining_sampling(clean_logo_png_path, dataset_dir, class_list, train_class_file, start_epoch, matrix_path, negative_only, gt):
	# get clean logo map 
	clean_logo_map = read_clean_logo_map(os.path.join(os.path.join(dataset_dir, clean_logo_png_path), 'clean_logo_map.npy'))
	# map of class id and class name, include class 0
	class_map = read_class_map(class_list)
	train_class_id_2_index, train_index_2_class_id = read_train_class(train_class_file)
	train_class_num = len(train_class_id_2_index)
	# build patch bins
	if gt == False:
		train_pos_patch_file = 'train_pos_list.txt'
	else:
		train_pos_patch_file = 'train_pos_gt_list.txt'
		
	train_neg_patch_file = os.path.join(os.path.join(dataset_dir, 'patches'), 'train_neg_list.txt')

	bins = build_patch_bins(os.path.join(os.path.join(dataset_dir, 'patches'), train_pos_patch_file), train_neg_patch_file)


	# initialize sampling matrix and vector as zero
	if start_epoch == 0:	
		# initialize sample matrix and vector
		neg_sample_matrix = np.zeros((train_class_num+1, train_class_num+1))
		neg_sample_matrix = torch.from_numpy(neg_sample_matrix)
		if negative_only:
			pos_sample_vector = []
		else:	
			pos_sample_vector = np.zeros((1,train_class_num))
			pos_sample_vector = torch.from_numpy(pos_sample_vector)

		print("Initialized sampling matrix and vector as 0.")
	# resume	
	else:
		# load sample matrix and vector	
		neg_sample_matrix = torch.from_numpy(np.load(os.path.join(matrix_path, "neg_sample_matrix_"+str(start_epoch)+".npy")))
		if negative_only:
			pos_sample_vector = []
		else:	
			pos_sample_vector = torch.from_numpy(np.load(os.path.join(matrix_path, "pos_sample_vector_"+str(start_epoch)+".npy")))

		print("Loaded sampling matrix and vector at epoch "+str(start_epoch))

	return class_map, clean_logo_map, pos_sample_vector, neg_sample_matrix, bins, train_class_id_2_index, train_index_2_class_id

# dynamic data loader
def get_dynamic_train_loader(class_list, dataset_path, patch_dir, clean_logo_path, 
			logo_rate, clean_logo_map, logo_bins, no_logo_list, logo_counts, no_logo_counts, class_map, 
			fixed_logo_bins, round_i, sample_path, train_transform, standard_transform, my_batch_size, load_num_workers, val, gt):
	print("==> Negative sampling, round "+str(round_i))
	train_neg_list, logo_bins, no_logo_list = generate_neg_pairs(class_list, dataset_path, patch_dir, clean_logo_path, 
			logo_rate, clean_logo_map, logo_bins, no_logo_list, logo_counts, no_logo_counts, class_map, fixed_logo_bins, 'train')
	# print(train_neg_list[0])
	# print(train_neg_list[1000])
	# print(train_neg_list[5000])
	# print(train_neg_list[-1])
	print("==> Sampled negative samples: "+str(len(train_neg_list)))
	dump_list(train_neg_list, round_i, sample_path, "neg")
	print("==> Saved negative samples.")

	if val == True:
		pair_file = "pairs_val/train_pos.txt"
	elif gt == True:
		pair_file = "pairs_gt/train_pos.txt"
	else:
		pair_file = "pairs/train_pos.txt"

	dynamic_train_dataset_instance = ImagePairDatasetDynamic(root=dataset_path, 
		pos_flist=os.path.join(dataset_path, pair_file),
		neg_list=train_neg_list,
		source_transform=train_transform,
		target_transform=standard_transform)
	
	dynamic_train_loader = torch.utils.data.DataLoader(
		dynamic_train_dataset_instance,
		batch_size=my_batch_size, shuffle=True,
		num_workers=load_num_workers, pin_memory=True)

	return dynamic_train_loader, logo_bins, no_logo_list


# hard mining data loader
def get_hard_mining_train_loader(patch_dir, dataset_path, clean_logo_path, class_map, clean_logo_map, 
	pos_sample_vector, neg_sample_matrix, bins, round_i, sample_path, train_transform, standard_transform, my_batch_size,
	load_num_workers, matrix_path, negative_only, val, gt):
	

	# load prepared set for epoch 0
	if round_i == 0: 
		if val == True:
			pair_file = "pairs_val/train_hm.txt"
		elif gt == True:
			pair_file = "pairs_gt/train_hm.txt"
		else:
			pair_file = "pairs/train_hm.txt"

		hm_train_dataset_instance = ImagePairDatasetHM(root=dataset_path, 
		flist=os.path.join(dataset_path, pair_file),
		source_transform=train_transform,
		target_transform=standard_transform)

		train_list = hm_train_dataset_instance.get_imlist()
		print("==> Round "+str(round_i)+": read " + str(len(train_list)) + " training pairs from file.")
		neg_sample_matrix, pos_sample_vector = init_sampling_matrix_vector(train_list, neg_sample_matrix, pos_sample_vector, 
			train_class_id_2_index, negative_only)
		print("Initialize sampling matrix and vector based on loaded training pairs.")

		# save initial sample matrix
		np.savetxt(os.path.join(matrix_path, "neg_sample_matrix_0.txt"), neg_sample_matrix.numpy(), fmt='%d')
		np.save(os.path.join(matrix_path, "neg_sample_matrix_0.npy"), neg_sample_matrix.numpy())
		if negative_only == False:
			np.savetxt(os.path.join(matrix_path, "pos_sample_vector_0.txt"), pos_sample_vector.numpy(), fmt='%d')
			np.save(os.path.join(matrix_path, "pos_sample_vector_0.npy"), pos_sample_vector.numpy())

		print("Saved them.")
		

	# sample for other round	
	else:
		print("==> Hard mining, round "+str(round_i))
		
		neg_pairs = generate_negative_pairs(dataset_path, patch_dir, clean_logo_path, class_map, neg_sample_matrix, bins, train_index_2_class_id)
		dump_list(neg_pairs, round_i, sample_path, "neg")
		if negative_only == False:
			pos_pairs = generate_positive_pairs(dataset_path, patch_dir, clean_logo_path, class_map, clean_logo_map, pos_sample_vector, bins, train_index_2_class_id)
			dump_list(pos_pairs, round_i, sample_path, "pos")
			print("==> Saved negative and positive samples.")
		else:
			print("==> Saved negative samples.")	

		if negative_only == False:
			hm_train_dataset_instance = ImagePairDatasetDynamicHM(root=dataset_path,
			pos_list=pos_pairs, neg_list=neg_pairs,
			source_transform=train_transform,
			target_transform=standard_transform)
		else:
			if val == True:
				pos_pair_file = "pairs_val/train_pos_hm.txt"
			elif gt == True:
				pos_pair_file = "pairs_gt/train_pos_hm.txt"
			else:
				pos_pair_file = "pairs/train_pos_hm.txt"

			hm_train_dataset_instance = ImagePairDatasetHM_Neg(root=dataset_path, 
			pos_flist=os.path.join(dataset_path, pos_pair_file),
			neg_list=neg_pairs,
			source_transform=train_transform,
			target_transform=standard_transform)
		

	hm_train_loader = torch.utils.data.DataLoader(
		hm_train_dataset_instance,
		batch_size=my_batch_size, shuffle=True,
		num_workers=load_num_workers, pin_memory=True)

	return hm_train_loader, neg_sample_matrix, pos_sample_vector


# save checkpoint
def save_checkpoint(state, path, cur_epoch):
	filename = "checkpoint_" + str(cur_epoch)+".pth"
	torch.save(state, os.path.join(path, filename))



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

# adjust learning rate
def adjust_learning_rate(optimizer, epoch, step, lr_dec):
	"""Decrease the learning rate by 0.1 every step epochs"""
	#if (step == 1 and epoch != 0) or (step != 1 and epoch != 0 and (epoch % step == 0)):
	if (epoch != 0) and (epoch % step == 0):
		print("Decrease the learning rate by "+str(lr_dec)+", for "+str(len(optimizer.param_groups))+" parameter groups.")
		for param_group in optimizer.param_groups:
			# assume epoch starts from 0
			param_group['lr'] = param_group['lr'] * lr_dec


# compute accuracy
def accuracy(output, target):
	"""Computes the precision@k for the specified values of k"""
	batch_size = target.size(0)
	# output: B*2
	# pred: B*1
	# choose the larger one for each row
	# the reason we can do this is because in softmax, exp is monotone
	_, pred = output.topk(1, 1, True, True)
	# pred: 1*B, indices of 0 or 1
	pred = pred.t()
	# correct: 1*B
	# target.view(1, -1): 1*B
	correct = pred.eq(target.view(1, -1))
	# correct_1: scalar
	correct_1 = correct.view(-1).float().sum(0)
	res = correct_1.mul_(100.0 / batch_size)
	return res

# return accuracy with predicted labels
def accuracy_pred(output, target):
	"""Computes the precision@k for the specified values of k"""
	batch_size = target.size(0)
	# output: B*2
	# pred: B*1
	# choose the larger one for each row
	# the reason we can do this is because in softmax, exp is monotone
	_, pred = output.topk(1, 1, True, True)
	# pred: 1*B, indices of 0 or 1
	pred = pred.t()
	# correct: 1*B
	# target.view(1, -1): 1*B
	correct = pred.eq(target.view(1, -1))
	# correct_1: scalar
	correct_1 = correct.view(-1).float().sum(0)
	res = correct_1.mul_(100.0 / batch_size)
	return res, pred.squeeze()

# dump list to a file
def dump_list(l, round_i, sample_path, spec):
	with open(os.path.join(sample_path, spec+"_"+str(round_i)+".txt"), 'w') as f:
		n = len(l[0])
		for line in l:
			s = ""
			for i in list(range(n)):
				s += str(line[i])
				if i < n-1:
					s += " "
				else:	
					s += "\n"
			
			f.write(s)


# train for one epoch
def train(train_loader, model, criterion, optimizer, epoch, print_freq, model_name, online_hard_mining=False):
	batch_time = AverageMeter()
	#data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	for i, (source_imgs, target_imgs, labels) in enumerate(train_loader):
	
		# measure data loading time
		#data_time.update(time.time() - end)

		labels = labels.cuda(async=True)
		if "siamese_contrastive" in model_name:
			label_var = torch.autograd.Variable(labels.float())
		else:
			label_var = torch.autograd.Variable(labels)

		#label_var size: [batch_size]	
				
		source_var = torch.autograd.Variable(source_imgs).cuda()
		target_var = torch.autograd.Variable(target_imgs).cuda()
		
		# compute output
		if "_stn" in model_name:
			if "_double" in model_name:
				small_source_imgs, small_target_imgs = get_downsampled_images(source_imgs, target_imgs)
				small_source_var = torch.autograd.Variable(small_source_imgs)
				small_target_var = torch.autograd.Variable(small_target_imgs)
				output = model(source_var, target_var, small_source_var, small_target_var)
			elif "_single" in model_name:
				small_source_imgs = get_downsampled_images_source(source_imgs)
				small_source_var = torch.autograd.Variable(small_source_imgs)
				output = model(source_var, target_var, small_source_var)
			elif "_late" in model_name:
				output = model(source_var, target_var)		
		else:	
			if "siamese_contrastive" not in model_name:
				# feed forward
				output = model(source_var, target_var)
			else:
				source_output, target_output = model(source_var, target_var)	
		
		
		# measure accuracy
		# prec1 has shape torch.Size([1])
		if "siamese_contrastive" not in model_name:
			prec1 = accuracy(output.data, labels)
			top1.update(prec1[0], source_imgs.size(0))
		
		# compute loss and backprop
		if online_hard_mining == False:
			if "siamese_contrastive" not in model_name:
				# loss.data has shape torch.Size([1])
				loss = criterion(output, label_var)
			else:
				loss = criterion(source_output, target_output, label_var)	
			#sys.exit()
			#print(loss.data[0])
			losses.update(loss.data[0], source_imgs.size(0))
			optimizer.zero_grad()
			loss.backward()
		### add hard negative mining here
		else:
			# pick those violate most	
			#m = Select_Loss()
			#loss_sub = pick_hard_samples(output.data, label_var)
			#loss_sub = m(output, label_var)
			m = ElementCrossEntropyLoss()
			loss_sub = m.pick_loss(output, label_var)
			#print(output.size())
			#print(label_var.size())
			print(loss_sub)
			optimizer.zero_grad()
			#loss_sub.backward()

		# do SGD step
		optimizer.step()
		
		
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# verbose
		if i % print_freq == 0:
			if "siamese_contrastive" not in model_name:
			# loss accuracy is computed at currrent batch (epoch average)
				print('Epoch: [{0}][{1}/{2}]\n'
					'\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
					'\tLoss: {loss.val:.4f} ({loss.avg:.4f})\n'
					'\tAccuracy: {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					loss=losses, top1=top1))
			else:
				print('Epoch: [{0}][{1}/{2}]\n'
					'\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
					'\tLoss: {loss.val:.4f} ({loss.avg:.4f})\n'.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					loss=losses))	


		#break	
				
	return losses.avg, top1.avg
		

# train for one epoch
'''
def train_online_hm(pos_train_loader, model, criterion, optimizer, epoch, print_freq, model_name):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	# read one pos pair
	for i, (source_imgs, target_imgs, labels) in enumerate(pos_train_loader):
		labels = labels.cuda(async=True)
		label_var = torch.autograd.Variable(labels)
				
		source_var = torch.autograd.Variable(source_imgs).cuda()
		target_var = torch.autograd.Variable(target_imgs).cuda()
		
		# feed forward
		output = model(source_var, target_var)
			
		
		# measure accuracy
		prec1 = accuracy(output.data, labels)
		top1.update(prec1[0], source_imgs.size(0))
		
		
		loss = criterion(output, label_var)
		losses.update(loss.data[0], source_imgs.size(0))
		optimizer.zero_grad()
		loss.backward()
		

		# do SGD step
		optimizer.step()
		
		
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# verbose
		if i % print_freq == 0:
			# loss accuracy is computed at currrent batch (epoch average)
			print('Epoch: [{0}][{1}/{2}]\n'
				'\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
				'\tLoss: {loss.val:.4f} ({loss.avg:.4f})\n'
				'\tAccuracy: {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				loss=losses, top1=top1))


		#break	
				
	return losses.avg, top1.avg	
'''		

# train for one epoch with offline hard mining
def train_hm(train_loader, model, criterion, optimizer, epoch, print_freq, model_name, neg_correct_matrix, 
	pos_correct_vector, neg_sample_matrix, pos_sample_vector, matrix_path, train_class_id_2_index, negative_only, unbalanced):

	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	for i, (source_imgs, target_imgs, labels, src_cls_ids, tgt_cls_ids) in enumerate(train_loader):
	
		labels = labels.cuda(async=True)
		label_var = torch.autograd.Variable(labels)
		source_var = torch.autograd.Variable(source_imgs).cuda()
		target_var = torch.autograd.Variable(target_imgs).cuda()
		
		# compute output
		if "_stn" in model_name:
			if "_double" in model_name:
				small_source_imgs, small_target_imgs = get_downsampled_images(source_imgs, target_imgs)
				small_source_var = torch.autograd.Variable(small_source_imgs)
				small_target_var = torch.autograd.Variable(small_target_imgs)
				output = model(source_var, target_var, small_source_var, small_target_var)
			elif "_single" in model_name:
				small_source_imgs = get_downsampled_images_source(source_imgs)
				small_source_var = torch.autograd.Variable(small_source_imgs)
				output = model(source_var, target_var, small_source_var)	
		else:	
			# feed forward
			output = model(source_var, target_var)
		
		# evaluate
		# loss.data has shape torch.Size([1])

		loss = criterion(output, label_var)
		
		# measure accuracy and record loss
		# prec1 has shape torch.Size([1])
		prec1, pred = accuracy_pred(output.data, labels)

		# update accuracy matrix
		neg_correct_matrix, pos_correct_vector = update_class_accuracy(pred, labels, neg_correct_matrix, pos_correct_vector, src_cls_ids, 
			tgt_cls_ids, train_class_id_2_index, negative_only)

		# update loss and accuracy
		losses.update(loss.data[0], source_imgs.size(0))
		top1.update(prec1[0], source_imgs.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# verbose
		if i % print_freq == 0:
			# loss accuracy is computed at currrent batch (epoch average)
			print('Epoch: [{0}][{1}/{2}]\n'
				'\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
				'\tLoss: {loss.val:.4f} ({loss.avg:.4f})\n'
				'\tAccuracy: {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				loss=losses, top1=top1))
		
		# if i > 20:
		# 	break

	# compute sampling matrix
	#sample_n = int(pos_sample_vector.sum())
	if unbalanced == False:
		sample_n = int(neg_sample_matrix.sum())
	else:
		sample_n = int(neg_sample_matrix.sum()+pos_sample_vector.sum())

	print("End of epoch "+str(epoch)+": re-computing sampling matrix. N="+str(sample_n))		
	# re-sample
	neg_sample_matrix, pos_sample_vector, neg_accuracy_matrix, pos_accuracy_vector = get_sampling_strategy(neg_correct_matrix, neg_sample_matrix, pos_correct_vector, pos_sample_vector, sample_n, negative_only, unbalanced)
	# save related matrices
	save_correct_sample_matrix(neg_accuracy_matrix, neg_sample_matrix, neg_correct_matrix, pos_accuracy_vector, pos_sample_vector, pos_correct_vector, matrix_path, epoch+1, negative_only)		
	
	print("Saved correct, accuracy and sampling matrix and vector for epoch "+str(epoch+1)+".")	

	return losses.avg, top1.avg, neg_correct_matrix, pos_correct_vector, neg_sample_matrix, pos_sample_vector

###  main
if __name__ == '__main__':
	
	###  arguments (store_true already created false as default)
	parser = argparse.ArgumentParser(description='PyTorch Pair VGG Training')
	parser.add_argument('--resume', action='store_true', help='resume from specified checkpoint (default: none)')
	parser.add_argument('--resume_epoch', default=0, type=int, help='resume to checkpoint number')
	parser.add_argument('--finetune', action='store_true', help='finetune the model based on Imagenet')
	parser.add_argument('--freeze', action='store_true', help='freeze the pretrained layers')
	parser.add_argument('--end2end', action='store_true', help='finetune the model using results from previous step')
	parser.add_argument('--model', default='vgg16_stack', type=str, help='name of model architecture')
	parser.add_argument('--lr', default=0.1, type=float, help='base learning rate')
	parser.add_argument('--lr_rate', default=0.1, type=float, help='ratio of learning rate between pretrained layers and new layers')
	parser.add_argument('--lr_dec', default=0.1, type=float, help='decrement multiplier of the learning rate')
	parser.add_argument('--dynamic', action='store_true', help='dynamically generate negative pairs.')
	parser.add_argument('--hard_mining', action='store_true', help='mininig hard pairs for each epoch.')
	parser.add_argument('--negative_only', action='store_true', help='mininig only hard negative pairs for each epoch.')
	parser.add_argument('--epochs', default=4, type=int, help='total number of epochs')
	parser.add_argument('--lr_drop_step', default=1, type=int, help='step of decreasing the learning rate')
	parser.add_argument('--batch_size', default=128, type=int, help='batch size')
	parser.add_argument('--unbalanced', action='store_true', help='during hard mining, do not keep balance between positives and negatives.')

	args = parser.parse_args()

	###  hyper parameters
	use_cuda = torch.cuda.is_available()
	epochs = args.epochs # how many epochs do we want to train
	lr_drop_step = args.lr_drop_step # For how many epoches should we decrease the learning rate
	momentum = 0.9
	weight_decay = 5e-4
	base_lr = args.lr
	lr_rate = args.lr_rate # ratio of learning rate between finetune old layer and new layer
	lr_dec = args.lr_dec 

	input_size = 224  # resize input image to square
	my_batch_size = args.batch_size # 256 will lead to out of memory

	print_freq = 20
	load_num_workers = 32

	###  Paths
	if args.end2end:
		model_data_dir = os.path.join(work_dir, "exps_end2end")
	else:	
		model_data_dir = os.path.join(work_dir, "exps")

	if not os.path.exists(model_data_dir):
		os.makedirs(model_data_dir)

	# model path (w or w/o dynamic sampling, hard mininig)
	model_path = os.path.join(model_data_dir, args.model)

	if args.dynamic:
		logo_rate = 0.5
		#lr_drop_step = 2*lr_drop_step
		model_path += "_dynamic"
		print("Using dynamic negative pair sampling: logo_rate="+str(logo_rate)+", lr_drop_step="+str(lr_drop_step))
	if args.hard_mining:
		model_path += "_hm"
		print("Using hard mining")
		if args.negative_only:
			model_path += "_neg"
			print("Hard mining for negatives only.")
		if args.unbalanced:
			model_path += "_unbalanced"
			print("Hard mining with unbalanced positives and negatives.")
						
	
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	# experimental setting	
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
	# resume path	
	resume_path = os.path.join(checkpoint_path, "checkpoint_" + str(args.resume_epoch) + ".pth")
	
	# dynamic sample path
	if args.hard_mining or args.dynamic:
		sample_path = os.path.join(exp_path, "samples") 
		if not os.path.exists(sample_path):
			os.makedirs(sample_path)

	# matrix path for hard mining 
	if args.hard_mining:		
		matrix_path = os.path.join(exp_path, "matrix") 
		if not os.path.exists(matrix_path):
			os.makedirs(matrix_path)

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
	

	# log path
	log_path = os.path.join(exp_path, "logs") 
	if not os.path.exists(log_path):
		os.makedirs(log_path)

	### redirect standard output	
	### name log file as current time
	current_time = time.time()
	st = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
	logfile = open(os.path.join(log_path,'train_'+st+'.txt'), 'w')
	sys.stdout = Logger(sys.stdout, logfile)


	### print out arguments
	print('==> Basic settings...')
	print('batch size: '+str(my_batch_size))
	print('total epochs: '+str(epochs))
	print('base lr: '+str(base_lr))
	print('lr drop step: '+str(lr_drop_step))

	if "_bn" in args.model :
		print('Using batch normalization')
	else:
		print('DO NOT use batch normalization')	

	### data loading
	mean_R, mean_G, mean_B = read_mean_image(os.path.join(train_dir, "mean.txt"))

	print('==> Preparing data...')
	standard_transform = transforms.Compose(
		[transforms.Scale((input_size,input_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[mean_R, mean_G, mean_B], std=[1,1,1])
		])

	if random.random() < 0.5:
		train_transform = transforms.Compose(
			[transforms.Scale((input_size,input_size)),
			transforms.ToTensor(),
			RandomAffine(rotation_range=90, shear_range=30, zoom_range=(0.8,1.2)),
			transforms.Normalize(mean=[mean_R, mean_G, mean_B], std=[1,1,1])
			])
	else:
		train_transform = standard_transform
	

	train_dataset_instance = ImagePairDataset(root=train_dir, 
		flist=os.path.join(train_pair_dir, "train_all.txt"),
		source_transform=train_transform,
		target_transform=standard_transform)
	
	train_loader = torch.utils.data.DataLoader(
		train_dataset_instance,
		batch_size=my_batch_size, shuffle=True,
		num_workers=load_num_workers, pin_memory=True)


	### Create a model
	print("==> Creating raw model...")
	# start from epoch 0
	start_epoch = 0 
	# create model and optimizer
	if "siamese_contrastive" not in args.model:
		model, optimizer = create_model_optimizer(model_name=args.model, pretrained=args.finetune, 
				pretrain_path=pretrain_path, base_lr=base_lr, freeze=args.freeze, 
				lr_rate=lr_rate, momentum=momentum, weight_decay=weight_decay)
		print("Using SGD optimizer")
	else:
		model, optimizer = create_model_optimizer_contrastive(model_name=args.model, pretrained=args.finetune, 
				pretrain_path=pretrain_path, freeze=args.freeze, lr_rate=lr_rate, base_lr=0.0005)
		print("Using Adam optimizer")	

	### define loss function (criterion)
	# cross entropy loss is from 0 to inf
	if "siamese_contrastive" in args.model:
		criterion = ContrastiveLoss()
		print("Using contrastive loss")
	elif "_focal" in args.model:
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
		#gpu_usage = [0,1,2,3,4]
		#gpu_usage = [0,1,2,3,4,5,6,7]
		#gpu_usage = [0,1,2]
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

	# resume model and optimizer from a checkpoint (should happen afer data parallel)	
	if args.resume:
		if os.path.isfile(resume_path):
			print("=> Resuming from checkpoint...")
			print("=> Loading checkpoint '{}'".format(resume_path))
			# load checkpoint
			checkpoint = torch.load(resume_path)
			# start from checkpoint epoch + 1
			start_epoch = checkpoint['epoch'] + 1
			# load model
			model.load_state_dict(checkpoint['state_dict'])
			# load optimizer
			optimizer.load_state_dict(checkpoint['optimizer'])
			start_loss_avg = checkpoint['loss_avg']
		
			print("=> Loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
			print("Avg loss of resummed epoch: %.3f" %(start_loss_avg))
			if "siamese_contrastive" not in args.model:
				start_acc_avg = checkpoint['accuracy_avg']
				print("Avg training accuracy of resummed epoch: %.3f%%" %(start_acc_avg))
		else:
			print("=> No checkpoint found at '{}'".format(resume_path))
	

	###  training
	print('==> Start training from epoch ' + str(start_epoch) + '...')
	start_time = time.time()
	start_st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M:%S')
	print('==> Start time: '+start_st)

	# prepare for dynamic negative sampling
	if args.dynamic:
		class_list = os.path.join(dataset_path, 'split/class_map.npy')
		patch_dir = 'patches'
		clean_logo_path = 'clean_logos'
		clean_logo_png_path = 'clean_logos_png'
		# do not consider args.val==True
		clean_logo_map, logo_bins, fixed_logo_bins, logo_counts, no_logo_counts, no_logo_list, class_map \
		= prepare_dynamic_negative_sampling(dataset_path, clean_logo_png_path, patch_dir, class_list, args.gt)
	elif args.hard_mining:
		class_list = os.path.join(dataset_path, 'split/class_map.npy')
		patch_dir = 'patches'
		clean_logo_png_path = 'clean_logos_png'
		clean_logo_path = 'clean_logos'
		train_class_file = os.path.join(os.path.join(dataset_path, "split"), "train_class.txt")

		# initialize or resume
		# do not consider args.val==True
		class_map, clean_logo_map, pos_sample_vector, neg_sample_matrix, \
		bins, train_class_id_2_index, train_index_2_class_id \
		= prepare_hard_mining_sampling(clean_logo_png_path, dataset_path, class_list, train_class_file, start_epoch, matrix_path, args.negative_only, args.gt)

	
	round_i = start_epoch
	for epoch in list(range(start_epoch, epochs)):
		# Start Time
		cur_time = time.time()
		cur_st = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d_%H:%M:%S')
		print('==> Epoch '+str(epoch)+': '+cur_st)
		
		if "siamese_contrastive" not in args.model:
			# adjust learning rate (at the beginning, so use epoch instead of epoch+1)
			adjust_learning_rate(optimizer, epoch, step=lr_drop_step, lr_dec=lr_dec)
		
		# train for one epoch
		# model, optimizer are passed by reference, not by value (according to imagenet example)
		if args.dynamic:
			dynamic_train_loader, logo_bins, no_logo_list = get_dynamic_train_loader(class_list, dataset_path, patch_dir, clean_logo_path, 
			logo_rate, clean_logo_map, logo_bins, no_logo_list, logo_counts, no_logo_counts, class_map, fixed_logo_bins, round_i, sample_path, 
			train_transform, standard_transform, my_batch_size, load_num_workers, args.val, args.gt)
			
			loss_avg, acc_avg = train(dynamic_train_loader, model, criterion, optimizer, epoch, print_freq, args.model)

		elif args.hard_mining:
			neg_correct_matrix, pos_correct_vector = init_correct_matrix_vector(train_class_id_2_index, args.negative_only)
			hm_train_loader, neg_sample_matrix, pos_sample_vector = get_hard_mining_train_loader(patch_dir, dataset_path, clean_logo_path, class_map, 
				clean_logo_map, pos_sample_vector, neg_sample_matrix, bins, round_i, 
				sample_path, train_transform, standard_transform, my_batch_size, 
				load_num_workers, matrix_path, args.negative_only, args.val, args.gt)
			
			loss_avg, acc_avg, neg_correct_matrix, pos_correct_vector, neg_sample_matrix, pos_sample_vector = train_hm(hm_train_loader, model, criterion, 
				optimizer, epoch, print_freq, args.model, neg_correct_matrix, 
				pos_correct_vector, neg_sample_matrix, pos_sample_vector, matrix_path, train_class_id_2_index, args.negative_only, args.unbalanced)
			
		elif args.model == "siamese_softmax_bn":
			loss_avg, acc_avg = train(train_loader, model, criterion, optimizer, epoch, print_freq, args.model, online_hard_mining=True)
		
		else:
			loss_avg, acc_avg = train(train_loader, model, criterion, optimizer, epoch, print_freq, args.model)

		# save current checkpoint
		# Note that optimizer can be saved only after iterating over at least one batch
		# otherwise when loading its state_dict is different from saved
		
		
		print("Saving checkpoint...")
		if "siamese_contrastive" not in args.model:
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'loss_avg': loss_avg,
				'accuracy_avg': acc_avg
			}, path=checkpoint_path, cur_epoch=epoch)
		else:
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'loss_avg': loss_avg
			}, path=checkpoint_path, cur_epoch=epoch)	
		
		
		# End time
		print('==> Epoch '+str(epoch)+': '+str(datetime.timedelta(seconds=time.time()-cur_time)))

		# for dynamic or hard mining
		round_i += 1

		#if epoch > 2:
		#	break
	
		

	end_time = time.time()
	end_st = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d_%H:%M:%S')
	print('==> End time: '+end_st)
	print('==> Total training time: '+str(datetime.timedelta(seconds=end_time-start_time)))
	print('Done.')
	


