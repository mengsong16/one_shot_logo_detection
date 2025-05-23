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

class Logger(object):
	def __init__(self, f1, f2):
		self.f1, self.f2 = f1, f2

	def write(self, msg):
		self.f1.write(msg)
		self.f2.write(msg)
		
	def flush(self):
		pass


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

# pred: 1*B
# target: B*1
# compute accuracy for pos and neg samples respectively
def pos_neg_accuracy(pred, target):
	batch_size = target.size(0)
	target = target.view(1, -1)
	
	assert target.size() == pred.size()
	# divide into pos and neg 

	pos_inds = [i for i in list(range(batch_size)) if target[0][i]==1]
	neg_inds = list((set(list(range(batch_size))) - set(pos_inds)))

	if len(pos_inds) == 0:
		pos_res = torch.FloatTensor([0])
	else:
		pos_target = torch.FloatTensor([target[0][i] for i in pos_inds])
		pos_pred = torch.FloatTensor([pred[0][i] for i in pos_inds])
		pos_correct = pos_pred.eq(pos_target)
		pos_correct_1 = pos_correct.view(-1).float().sum(0)
		pos_res = pos_correct_1.mul_(100.0 / len(pos_inds))

	if len(neg_inds) == 0:
		neg_res = torch.FloatTensor([0])	
	else:
		neg_target = torch.FloatTensor([target[0][i] for i in neg_inds])
		neg_pred = torch.FloatTensor([pred[0][i] for i in neg_inds])
		neg_correct = neg_pred.eq(neg_target)
		neg_correct_1 = neg_correct.view(-1).float().sum(0)
		neg_res = neg_correct_1.mul_(100.0 / len(neg_inds))


	return pos_res, neg_res, len(pos_inds), len(neg_inds)

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

# test (no hard mining)
def validate(val_loader, model, criterion, print_freq, model_name, epoch):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	pos_top1 = AverageMeter()
	neg_top1 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (source_imgs, target_imgs, labels) in enumerate(val_loader):
		#print(str(i))
		#if (i+1) % 100 == 0:
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
			elif "_late" in model_name:
				output = model(source_var, target_var)		
		else:
			output = model(source_var, target_var)

		loss = criterion(output, label_var)

		# measure accuracy and record loss
		prec1, pred = accuracy(output.data, labels)
		pos_prec1, neg_prec1, pos_size, neg_size = pos_neg_accuracy(pred, labels)
		#print(prec1)
		losses.update(loss.data[0], source_imgs.size(0))
		top1.update(prec1[0], source_imgs.size(0))
		if pos_size > 0:
			pos_top1.update(pos_prec1[0], pos_size)
		if neg_size > 0:	
			neg_top1.update(neg_prec1[0], neg_size)

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			print('Test: [Epoch: {2}][{0}/{1}]\n'
				'\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'\tLoss: {loss.val:.4f} ({loss.avg:.4f})\n'
				'\tAccuracy: {top1.val:.3f}% ({top1.avg:.3f}%)\n'
				'\tPos Accuracy: {pos_top1.val:.3f}% ({pos_top1.avg:.3f}%)\n'
				'\tNeg Accuracy: {neg_top1.val:.3f}% ({neg_top1.avg:.3f}%)\n'.format(
				i, len(val_loader), epoch, batch_time=batch_time, loss=losses,
				top1=top1, pos_top1=pos_top1, neg_top1=neg_top1))

		#break	

	print(' * Accuracy: {top1.avg:.3f}%'.format(top1=top1))
	print(' * Pos Accuracy: {pos_top1.avg:.3f}%'.format(pos_top1=pos_top1))
	print(' * Neg Accuracy: {neg_top1.avg:.3f}%'.format(neg_top1=neg_top1))

	return top1.avg, pos_top1.avg, neg_top1.avg

# compute accuracy for pair of (classA, classB)
def compute_class_accuracy(neg_correct_matrix, neg_num_matrix, pos_correct_vector, pos_num_vector):
	
	neg_correct_matrix = neg_correct_matrix.float()
	neg_num_matrix = neg_num_matrix.float()
	pos_correct_vector = pos_correct_vector.float()
	pos_num_vector = pos_num_vector.float()

	# initialize results as zeros
	neg_accuracy_matrix = torch.zeros(neg_num_matrix.size()).float()
	pos_accuracy_vector = torch.zeros(pos_num_vector.size()).float()

	rn = neg_num_matrix.size()[0]
	cn = neg_num_matrix.size()[1]
	ccn = pos_num_vector.size()[1]
	
	# compute accuracy
	for i in list(range(rn)):
		for j in list(range(cn)):
			if neg_num_matrix[i][j] > 0:
				assert neg_num_matrix[i][j] >= neg_correct_matrix[i][j]
				neg_accuracy_matrix[i][j] = neg_correct_matrix[i][j] / neg_num_matrix[i][j]

	for i in list(range(ccn)):
		if pos_num_vector[0][i] > 0:
			assert pos_num_vector[0][i] >= pos_correct_vector[0][i]
			pos_accuracy_vector[0][i] = pos_correct_vector[0][i] / pos_num_vector[0][i]			
	
	return neg_accuracy_matrix, pos_accuracy_vector

# initialize vector and matrix as 0
def init_correct_matrix_vector_as_0(class_id_2_index_map):

	print("==> Initialize counting matrix and vector as 0.")
	class_num = len(class_id_2_index_map)
	neg_sample_matrix = np.zeros((class_num+1, class_num+1))
	pos_sample_vector = np.zeros((1,class_num))
	neg_sample_matrix = torch.from_numpy(neg_sample_matrix)
	pos_sample_vector = torch.from_numpy(pos_sample_vector)


	return neg_sample_matrix, pos_sample_vector

# test with computing class accuracy (for hard mining)
def validate_class(val_loader, model, criterion, print_freq, model_name, test_class_id_2_index, epoch):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	pos_top1 = AverageMeter()
	neg_top1 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	# initialize correct matrix and vector
	neg_correct_matrix, pos_correct_vector = init_correct_matrix_vector_as_0(test_class_id_2_index)

	

	# start test over the whole testset
	end = time.time()
	for i, (source_imgs, target_imgs, labels, src_cls_ids, tgt_cls_ids) in enumerate(val_loader):
		#print(str(i))
		#if (i+1) % 100 == 0:
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
			elif "_late" in model_name:
				output = model(source_var, target_var)	
		else:
			output = model(source_var, target_var)

		loss = criterion(output, label_var)

		# measure accuracy and record loss
		prec1, pred = accuracy(output.data, labels)
		# update accuracy matrix
		neg_correct_matrix, pos_correct_vector = update_class_accuracy(pred.squeeze(), labels, neg_correct_matrix, pos_correct_vector, 
			src_cls_ids, tgt_cls_ids, test_class_id_2_index)
		# compute positive and negative accuracy
		pos_prec1, neg_prec1, pos_size, neg_size = pos_neg_accuracy(pred, labels)
		#print(prec1)
		losses.update(loss.data[0], source_imgs.size(0))
		top1.update(prec1[0], source_imgs.size(0))
		if pos_size > 0:
			pos_top1.update(pos_prec1[0], pos_size)
		if neg_size > 0:	
			neg_top1.update(neg_prec1[0], neg_size)

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			print('Test: [Epoch: {2}][{0}/{1}]\n'
				'\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'\tLoss: {loss.val:.4f} ({loss.avg:.4f})\n'
				'\tAccuracy: {top1.val:.3f}% ({top1.avg:.3f}%)\n'
				'\tPos Accuracy: {pos_top1.val:.3f}% ({pos_top1.avg:.3f}%)\n'
				'\tNeg Accuracy: {neg_top1.val:.3f}% ({neg_top1.avg:.3f}%)\n'.format(
				i, len(val_loader), epoch, batch_time=batch_time, loss=losses,
				top1=top1, pos_top1=pos_top1, neg_top1=neg_top1))

		#break	

	print(' * Accuracy: {top1.avg:.3f}%'.format(top1=top1))
	print(' * Pos Accuracy: {pos_top1.avg:.3f}%'.format(pos_top1=pos_top1))
	print(' * Neg Accuracy: {neg_top1.avg:.3f}%'.format(neg_top1=neg_top1))

	return top1.avg, pos_top1.avg, neg_top1.avg, neg_correct_matrix, pos_correct_vector	

# draw and save curve of training and test accuracy   
def draw_curve(train_epoch_list, test_epoch_list, train_acc_list, test_acc_list, test_acc_pos_list, test_acc_neg_list, output_file):
    train_h, = plt.plot(train_epoch_list, train_acc_list, 'r-*', label='train')
    test_h, = plt.plot(test_epoch_list, test_acc_list, 'b-*', label='test')
    test_pos_h, = plt.plot(test_epoch_list, test_acc_pos_list, 'g-*', label='test_pos')
    test_neg_h, = plt.plot(test_epoch_list, test_acc_neg_list, 'm-*', label='test_neg')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy Curves on Training / Test Set')
    plt.legend(handles=[train_h, test_h, test_pos_h, test_neg_h])
    plt.savefig(output_file)

def line2list(line):
	l = line.strip().split()
	
	return l

def read_results(test_file):
	with open(test_file, 'r') as f:
		line = f.readline()
		train_epoch_list = line2list(line)
		train_epoch_list = [float(x) for x in train_epoch_list]
		line = f.readline()
		test_epoch_list = line2list(line)
		test_epoch_list = [float(x) for x in test_epoch_list]
		line = f.readline()
		train_acc_list = line2list(line)
		train_acc_list = [float(x) for x in train_acc_list]
		line = f.readline()
		test_acc_list = line2list(line)
		test_acc_list = [float(x) for x in test_acc_list]
		line = f.readline()
		test_acc_pos_list = line2list(line)
		test_acc_pos_list = [float(x) for x in test_acc_pos_list]
		line = f.readline()
		test_acc_neg_list = line2list(line)
		test_acc_neg_list = [float(x) for x in test_acc_neg_list]
		 
	return 	train_epoch_list, test_epoch_list, train_acc_list, test_acc_list, test_acc_pos_list, test_acc_neg_list

def redraw_curve(test_file, test_pic):
	train_epoch_list, test_epoch_list, train_acc_list, test_acc_list, test_acc_pos_list, test_acc_neg_list = read_results(test_file)
	draw_curve(train_epoch_list, test_epoch_list, train_acc_list, test_acc_list, test_acc_pos_list, test_acc_neg_list, test_pic)

def list2str(L):
    s = " ".join(str(x) for x in L)
    s += "\n"
    return s

def class_id_to_matrix_index(class_id, class_id_2_index_map):
	if class_id == 0:
		return 0
	else:
		return class_id_2_index_map[class_id]

# intialize class accuracy vector and matrix based on the input list
def init_counting_matrix_vector(data_list, class_id_2_index_map):
	neg_sample_matrix, pos_sample_vector = init_correct_matrix_vector_as_0(class_id_2_index_map)

	print("==> Initialize counting matrix and vector based on loaded pairs.")
	for item in data_list:
		label = item[2]
		src_cls_id = item[3]
		tgt_cls_id = item[4]
		if label == 1:
			ind = class_id_to_matrix_index(src_cls_id, class_id_2_index_map) - 1
			pos_sample_vector[0][ind] += 1
		else:
			col = class_id_to_matrix_index(src_cls_id, class_id_2_index_map)
			row = class_id_to_matrix_index(tgt_cls_id, class_id_2_index_map)
			neg_sample_matrix[row][col] += 1	

	# check correctness
	n = 0
	vn = pos_sample_vector.size()[1]
	for i in list(range(vn)):
		n += pos_sample_vector[0][i]

	rn = neg_sample_matrix.size()[0]
	cn = neg_sample_matrix.size()[1]
		
	for j in list(range(rn)):
		for k in list(range(cn)):
			n += neg_sample_matrix[j][k]

	assert n == len(data_list)

	return neg_sample_matrix, pos_sample_vector


# map class id of testing classes to index starting from 1 and reverse
def read_test_class(class_file):	
	with open(class_file, 'r') as f:
		lines = f.readlines()

	lines = [x.strip() for x in lines]

	# class id => index from 1 
	index_2_class_id_map = {}
	class_id_2_index_map = {}
	
	i = 1
	for line in lines:
		class_id = int(line)
		class_id_2_index_map[class_id] = i
		index_2_class_id_map[i] = class_id
		i += 1

	return class_id_2_index_map, index_2_class_id_map
	

# compute accuracy of each class for negative pairs and positive pairs
def update_class_accuracy(pred, target, neg_correct_matrix, pos_correct_vector, src_cls_ids, tgt_cls_ids, train_class_id_2_index):
	batch_size = target.size(0)

	assert target.size() == pred.size()
	assert target.size() == src_cls_ids.size()
	assert target.size() == tgt_cls_ids.size()
	
	neg_inds = [i for i in list(range(batch_size)) if target[i]==0]
	pos_inds = list((set(list(range(batch_size))) - set(neg_inds)))


	if len(neg_inds) > 0:
		for i in neg_inds:
			if pred[i] == target[i]:
				col = class_id_to_matrix_index(src_cls_ids[i], train_class_id_2_index)
				row = class_id_to_matrix_index(tgt_cls_ids[i], train_class_id_2_index)
				neg_correct_matrix[row][col] += 1
				
	
	if 	len(pos_inds) > 0:
		for i in pos_inds:
			if pred[i] == target[i]:
				ind = class_id_to_matrix_index(src_cls_ids[i], train_class_id_2_index) - 1
				pos_correct_vector[0][ind] += 1
							

	return neg_correct_matrix, pos_correct_vector

###  main
if __name__ == '__main__':
	
	###  arguments
	parser = argparse.ArgumentParser(description='PyTorch Pair VGG Testing')
	parser.add_argument('--finetune', action='store_true', help='finetune the model based on Imagenet')
	parser.add_argument('--freeze', action='store_true', help='freeze the pretrained layers')
	parser.add_argument('--end2end', action='store_true', help='finetune the model using results from previous step')
	parser.add_argument('--model', default='vgg16_stack', type=str, help='name of model architecture')
	parser.add_argument('--dynamic', action='store_true', help='dynamically generate negative pairs')
	parser.add_argument('--classid', action='store_true', help='test accuracy for each class.')
	parser.add_argument('--hard_mining', action='store_true', help='mininig hard pairs for each epoch')
	parser.add_argument('--negative_only', action='store_true', help='mininig only hard negative pairs for each epoch.')
	parser.add_argument('--batch_size', default=128, type=int, help='batch size')
	parser.add_argument('--start_epoch', default=0, type=int, help='index of epoch where we start testing (from 0)')
	parser.add_argument('--end_epoch', default=2, type=int, help='index of epoch where we end testing (from 0)')
	parser.add_argument('--unbalanced', action='store_true', help='during hard mining, do not keep balance between positives and negatives.')
	parser.add_argument('--opt', default='test_w32', type=str, help='name of test set we want to use')
	
	args = parser.parse_args()
	
	###  hyper parameters
	use_cuda = torch.cuda.is_available()
	val_step = 1 # For how many epoches should we do validation
	input_size = 224  # resize input image to square
	my_batch_size = args.batch_size
	print_freq = 20
	load_num_workers = 32
	# start test from epoch x until y (indexed from 0)
	start_epoch = args.start_epoch
	end_epoch = args.end_epoch


	###  Paths
	# model data path
	if args.end2end:
		model_data_dir = os.path.join(work_dir, "exps_end2end")
	else:	
		model_data_dir = os.path.join(work_dir, "exps")

	if not os.path.exists(model_data_dir):
		os.makedirs(model_data_dir)

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

	
	# print out test options
	print('==> Test on set '+str(args.opt)+' ...')
	### print out arguments
	print('==> Basic settings...')
	print('batch size: '+str(my_batch_size))
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

	# test file path
	if args.opt == "test_w32":
		pair_file = os.path.join(test_pair_dir, "test_w32_all.txt")
		testset_path = test_dir
		class_file = os.path.join(csv_class_dir, "test_w32_class.txt")
	elif args.opt == "test_wo32":
		pair_file = os.path.join(test_pair_dir, "test_wo32_all.txt")
		testset_path = test_dir
		class_file = os.path.join(csv_class_dir, "test_wo32_class.txt")
	elif args.opt == "test_seen":
		pair_file = os.path.join(test_pair_dir, "test_seen_all.txt")
		testset_path = test_dir
		class_file = os.path.join(csv_class_dir, "test_seen_class.txt")
	elif args.opt == "val_unseen":
		pair_file = os.path.join(val_pair_dir, "val_all.txt")
		testset_path = val_dir	
		class_file = os.path.join(csv_class_dir, "val_class.txt")
	elif args.opt == "val_seen":
		pair_file = os.path.join(val_pair_dir, "val_seen_all.txt")
		testset_path = val_dir	
		class_file = os.path.join(csv_class_dir, "val_seen_class.txt")
	else:
		print("Error: test option is incorrect")	

	# with class accuracy		
	if args.classid:
		test_dataset_instance = ImagePairDatasetHM(root=testset_path, 
			flist=pair_file,
			source_transform=standard_transform,
			target_transform=standard_transform)

		# intilialize the accuracy matrix
		matrix_path = os.path.join(test_path, "matrix")
		if not os.path.exists(matrix_path):
			os.makedirs(matrix_path)

		class_id_2_index_map, _ = read_test_class(class_file)
		neg_num_matrix, pos_num_vector = init_counting_matrix_vector(test_dataset_instance.get_imlist(), class_id_2_index_map)
		# save counting matrix and vector
		np.savetxt(os.path.join(matrix_path, "neg_num_matrix.txt"), neg_num_matrix.numpy(), fmt='%d')	
		np.save(os.path.join(matrix_path, "neg_num_matrix.npy"), neg_num_matrix.numpy())
		np.savetxt(os.path.join(matrix_path, "pos_num_vector.txt"), pos_num_vector.numpy(), fmt='%d')	
		np.save(os.path.join(matrix_path, "pos_num_vector.npy"), pos_num_vector.numpy())

		

	# without class id	
	else:
		test_dataset_instance = ImagePairDataset(root=testset_path, 
			flist=pair_file,
			source_transform=standard_transform,
			target_transform=standard_transform)

	# data loader
	test_loader = torch.utils.data.DataLoader(
		test_dataset_instance,
		batch_size=my_batch_size, shuffle=False,
		num_workers=load_num_workers, pin_memory=True)

	### Create a model
	model = create_model(model_name=args.model, pretrained=args.finetune, pretrain_path=pretrain_path)
			
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
		#gpu_usage = [5,6,7]
		#gpu_usage = [0,1,2,3,4,5,6,7]
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

	
	###  testing
	# test on checkpoints
	best_prec1 = -1000000
	best_epoch = -2

	train_epoch_list = []
	test_epoch_list = []
	train_acc_list = []
	test_acc_list = []
	test_acc_pos_list = []
	test_acc_neg_list = []

	
	# If pretrained and test from beginning, test pretrained model
	#if False:
	if args.finetune == True and start_epoch == 0:
		epoch = -1
		cur_time = time.time()
		cur_st = datetime.datetime.fromtimestamp(cur_time).strftime('%Y-%m-%d_%H:%M:%S')
		print('==> Test on pretrained model, epoch '+str(epoch)+': '+cur_st)

		# pretrained model has already been loaded at this point
		# evaluate on test set
		# return avg accuracy
		if args.classid:
			prec1, pos_prec1, neg_prec1, neg_correct_matrix, pos_correct_vector = validate_class(test_loader, model, criterion, print_freq, args.model, class_id_2_index_map, epoch)
		else:
			prec1, pos_prec1, neg_prec1 = validate(test_loader, model, criterion, print_freq, args.model, epoch)
		#prec1 = random.uniform(0, 1)

		# update best prec@1 and save checkpoint
		if prec1 > best_prec1:
			best_prec1 = prec1
			best_epoch = epoch

		test_acc_list.append(prec1)
		test_epoch_list.append(epoch+1)
		test_acc_pos_list.append(pos_prec1)
		test_acc_neg_list.append(neg_prec1)

		# compute class accuracy matrix and save
		if args.classid:
			neg_accuracy_matrix, pos_accuracy_vector = compute_class_accuracy(neg_correct_matrix, neg_num_matrix, pos_correct_vector, pos_num_vector)
			np.savetxt(os.path.join(matrix_path, "neg_accuracy_matrix_"+str(epoch)+".txt"), neg_accuracy_matrix.numpy(), fmt='%.4f')
			np.savetxt(os.path.join(matrix_path, "pos_accuracy_vector_"+str(epoch)+".txt"), pos_accuracy_vector.numpy(), fmt='%.4f')
			np.save(os.path.join(matrix_path, "neg_accuracy_matrix_"+str(epoch)+".npy"), neg_accuracy_matrix.numpy())
			np.save(os.path.join(matrix_path, "pos_accuracy_vector_"+str(epoch)+".npy"), pos_accuracy_vector.numpy())

		# End time
		print('==> Pretrained model test Completed: '+str(datetime.timedelta(seconds=time.time()-cur_time)))
	

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
			model, _, acc_avg_train = resume_model_checkpoint(checkpoint_path, epoch, model)
			# evaluate on test set
			# return avg accuracy of this epoch
			if args.classid:
				prec1, pos_prec1, neg_prec1, neg_correct_matrix, pos_correct_vector = validate_class(test_loader, model, criterion, print_freq, args.model, class_id_2_index_map, epoch)
			else:	
				prec1, pos_prec1, neg_prec1 = validate(test_loader, model, criterion, print_freq, args.model, epoch)

			# update best prec@1 and save checkpoint
			if prec1 > best_prec1:
				best_prec1 = prec1
				best_epoch = epoch

			# save acc (epoch is indexed from -1)
			train_epoch_list.append(int(epoch+1))
			test_epoch_list.append(int(epoch+1))
			train_acc_list.append(acc_avg_train)
			test_acc_list.append(prec1)	
			test_acc_pos_list.append(pos_prec1)
			test_acc_neg_list.append(neg_prec1)

			# compute class accuracy matrix and save
			if args.classid:
				neg_accuracy_matrix, pos_accuracy_vector = compute_class_accuracy(neg_correct_matrix, neg_num_matrix, pos_correct_vector, pos_num_vector)
				np.savetxt(os.path.join(matrix_path, "neg_accuracy_matrix_"+str(epoch)+".txt"), neg_accuracy_matrix.numpy(), fmt='%.4f')
				np.savetxt(os.path.join(matrix_path, "pos_accuracy_vector_"+str(epoch)+".txt"), pos_accuracy_vector.numpy(), fmt='%.4f')
				np.save(os.path.join(matrix_path, "neg_accuracy_matrix_"+str(epoch)+".npy"), neg_accuracy_matrix.numpy())
				np.save(os.path.join(matrix_path, "pos_accuracy_vector_"+str(epoch)+".npy"), pos_accuracy_vector.numpy())
				
			# End time
			print('==> Epoch '+str(epoch)+': '+str(datetime.timedelta(seconds=time.time()-cur_time)))
		
		# next epoch
		i += 1
		#break

	# save acc
	with open(os.path.join(test_path,'acc_checkpoints.txt'), 'w') as f:
		f.write(list2str(train_epoch_list))
		f.write(list2str(test_epoch_list))
		f.write(list2str(train_acc_list))
		f.write(list2str(test_acc_list))
		f.write(list2str(test_acc_pos_list))
		f.write(list2str(test_acc_neg_list))

	# draw curve	
	draw_curve(train_epoch_list, test_epoch_list, train_acc_list, test_acc_list, test_acc_pos_list, test_acc_neg_list,
		os.path.join(test_path,'acc_curve_checkpoints.jpg'))
	
	# print summary
	end_time = time.time()
	end_st = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d_%H:%M:%S')
	print('==> End time: '+end_st)
	print('==> Total test time: '+str(datetime.timedelta(seconds=end_time-start_time)))
	print('Best epoch: '+str(best_epoch))
	print('Best test accuracy: %.3f%%' % (best_prec1))
	print('Done.')	

	#redraw_curve(os.path.join(test_path, "acc_checkpoints.txt"), os.path.join(test_path, "acc_curve_checkpoints.jpg"))

		
