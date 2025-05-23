import os
import sys
import math
import random
import shutil
import time
import numpy as np
import datetime
import torch

sys.path.append("../")
from config import *

def test_matrix(train_class_id_2_index):
	pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
	target = np.array([1, 0, 1, 0, 1, 0, 1, 0])
	src_cls_ids = np.array([0, 1, 1, 2, 2, 0, 0, 0])
	tgt_cls_ids = np.array([0, 0, 1, 0, 2, 2, 0, 1])
	neg_correct_matrix = np.zeros((3,3))
	pos_correct_vector = np.zeros((1,3))

	pred = torch.from_numpy(pred)
	target = torch.from_numpy(target)
	src_cls_ids = torch.from_numpy(src_cls_ids)
	tgt_cls_ids = torch.from_numpy(tgt_cls_ids)
	neg_correct_matrix = torch.from_numpy(neg_correct_matrix)
	pos_correct_vector = torch.from_numpy(pos_correct_vector)

	neg_correct_matrix, pos_correct_vector = update_class_accuracy(pred, target, neg_correct_matrix, pos_correct_vector, src_cls_ids, tgt_cls_ids, train_class_id_2_index)

	#print(neg_correct_matrix)
	#print(pos_correct_vector)

	neg_num_matrix = np.matrix([[0,1,1],[1,0,0],[1,0,0]])
	pos_num_vector = np.matrix([[2, 1, 1]])
	neg_num_matrix = torch.from_numpy(neg_num_matrix)
	pos_num_vector = torch.from_numpy(pos_num_vector)

	#print(neg_num_matrix)
	#print(pos_num_vector)

	neg_sample_matrix, pos_sample_vector = get_sampling_strategy(neg_correct_matrix, neg_num_matrix, pos_correct_vector, pos_num_vector, 100, unbalanced=False, eps=0.0001)

	print(neg_sample_matrix)
	print(pos_sample_vector)

def class_id_to_matrix_index(class_id, train_class_id_2_index):
	if class_id == 0:
		return 0
	else:
		return train_class_id_2_index[class_id]	

def matrix_index_to_class_id(matrix_ind, train_index_2_class_id):
	if matrix_ind == 0:
		return 0
	else:
		return train_index_2_class_id[matrix_ind]		

def init_correct_matrix_vector(train_class_id_2_index, negative_only):
	train_class_num = len(train_class_id_2_index)
	neg_correct_matrix = np.zeros((train_class_num+1, train_class_num+1))
	neg_correct_matrix = torch.from_numpy(neg_correct_matrix)

	if negative_only == True:
		pos_correct_vector = []
	else:	
		pos_correct_vector = np.zeros((1,train_class_num))
		pos_correct_vector = torch.from_numpy(pos_correct_vector)

	print("Initialized correct matrix and vector as 0.")

	return neg_correct_matrix, pos_correct_vector
	
# input zero matrix and vector
def init_sampling_matrix_vector(train_list, neg_sample_matrix, pos_sample_vector, train_class_id_2_index, negative_only):
	for item in train_list:
		label = item[2]
		src_cls_id = item[3]
		tgt_cls_id = item[4]
		if label == 1:
			if negative_only == False:
				ind = class_id_to_matrix_index(src_cls_id, train_class_id_2_index) - 1
				pos_sample_vector[0][ind] += 1
		else:
			col = class_id_to_matrix_index(src_cls_id, train_class_id_2_index)
			row = class_id_to_matrix_index(tgt_cls_id, train_class_id_2_index)
			neg_sample_matrix[row][col] += 1	

	# check correctness
	if negative_only == False:
		n = 0
		#print(pos_sample_vector.size())
		vn = pos_sample_vector.size()[1]
		for i in list(range(vn)):
			n += pos_sample_vector[0][i]

		rn = neg_sample_matrix.size()[0]
		cn = neg_sample_matrix.size()[1]
			
		for j in list(range(rn)):
			for k in list(range(cn)):
				n += neg_sample_matrix[j][k]

		assert n == len(train_list)		

	return neg_sample_matrix, pos_sample_vector

# compute accuracy of each class for negative pairs and positive pairs
def update_class_accuracy(pred, target, neg_correct_matrix, pos_correct_vector, src_cls_ids, tgt_cls_ids, train_class_id_2_index, negative_only):
	batch_size = target.size(0)

	assert target.size() == pred.size()
	assert target.size() == src_cls_ids.size()
	assert target.size() == tgt_cls_ids.size()

	#print(pred)
	#print(target)
	
	neg_inds = [i for i in list(range(batch_size)) if target[i]==0]
	if negative_only == False:
		pos_inds = list((set(list(range(batch_size))) - set(neg_inds)))


	if len(neg_inds) > 0:
		for i in neg_inds:
			if pred[i] == target[i]:
				col = class_id_to_matrix_index(src_cls_ids[i], train_class_id_2_index)
				row = class_id_to_matrix_index(tgt_cls_ids[i], train_class_id_2_index)
				neg_correct_matrix[row][col] += 1
				
	if negative_only == False:
		if 	len(pos_inds) > 0:
			for i in pos_inds:
				if pred[i] == target[i]:
					ind = class_id_to_matrix_index(src_cls_ids[i], train_class_id_2_index) - 1
					pos_correct_vector[0][ind] += 1
							

	return neg_correct_matrix, pos_correct_vector

def ind2rowcol(ind, cn):
	r = int(ind) // int(cn)
	c = int(ind) % int(cn)

	return r, c

def rowcol2ind(r, c, cn):
	return r*cn+c

def save_correct_sample_matrix(neg_accuracy_matrix, neg_sample_matrix, neg_correct_matrix, pos_accuracy_vector, pos_sample_vector, pos_correct_vector, matrix_path, epoch, negative_only):
	
	np.savetxt(os.path.join(matrix_path, "neg_accuracy_matrix_"+str(epoch)+".txt"), neg_accuracy_matrix.numpy(), fmt='%.4f')
	np.savetxt(os.path.join(matrix_path, "neg_sample_matrix_"+str(epoch)+".txt"), neg_sample_matrix.numpy(), fmt='%d')
	np.savetxt(os.path.join(matrix_path, "neg_correct_matrix_"+str(epoch)+".txt"), neg_correct_matrix.numpy(), fmt='%d')

	if negative_only == False:
		np.savetxt(os.path.join(matrix_path, "pos_accuracy_vector_"+str(epoch)+".txt"), pos_accuracy_vector.numpy(), fmt='%.4f')
		np.savetxt(os.path.join(matrix_path, "pos_sample_vector_"+str(epoch)+".txt"), pos_sample_vector.numpy(), fmt='%d')
		np.savetxt(os.path.join(matrix_path, "pos_correct_vector_"+str(epoch)+".txt"), pos_correct_vector.numpy(), fmt='%d')

	np.save(os.path.join(matrix_path, "neg_accuracy_matrix_"+str(epoch)+".npy"), neg_accuracy_matrix.numpy())
	np.save(os.path.join(matrix_path, "neg_sample_matrix_"+str(epoch)+".npy"), neg_sample_matrix.numpy())
	np.save(os.path.join(matrix_path, "neg_correct_matrix_"+str(epoch)+".npy"), neg_correct_matrix.numpy())

	if negative_only == False:
		np.save(os.path.join(matrix_path, "pos_accuracy_vector_"+str(epoch)+".npy"), pos_accuracy_vector.numpy())
		np.save(os.path.join(matrix_path, "pos_sample_vector_"+str(epoch)+".npy"), pos_sample_vector.numpy())
		np.save(os.path.join(matrix_path, "pos_correct_vector_"+str(epoch)+".npy"), pos_correct_vector.numpy())

	

# sample n patches from one bin
def sample_one_bin(bins, class_id, n):
	n = int(n)
	#print(class_id)
	if len(bins[class_id]) == n:
		return bins[class_id]
	elif len(bins[class_id]) > n:
		return random.sample(bins[class_id], n)	
	else:
		m = int(math.ceil(n / float(len(bins[class_id])))) - 1
		print("==> Over sampling: "+str(m+1) +" * [" +str(len(bins[class_id])) +"], n = "+str(n))
		assert m > 0
		#l = bins[class_id]
		#for i in list(range(m)):
		#	l += bins[class_id]
		l = bins[class_id] * m
		#print("Finish here")	
		#print("==>               pool size = "+str(l))
		#return random.sample(l, n)	
		l += random.sample(bins[class_id], n-len(l))
		assert len(l) == n
		return l

# neg_correct_matrix / neg_num_matrix: 267 * 267
# pos_correct_vector / pos_num_vector: 1 * 266
# N: number of positive / negative pairs
def get_sampling_strategy(neg_correct_matrix, neg_num_matrix, pos_correct_vector, pos_num_vector, N, negative_only, unbalanced, eps=0.0001):
	#assert neg_correct_matrix.size() == neg_num_matrix.size()
	#assert pos_correct_matrix.size() == pos_num_matrix.size()
	neg_correct_matrix = neg_correct_matrix.float()
	neg_num_matrix = neg_num_matrix.float()

	if negative_only == False:
		pos_correct_vector = pos_correct_vector.float()
		pos_num_vector = pos_num_vector.float()

	# initialize results as zeros
	neg_sample_matrix = torch.zeros(neg_num_matrix.size()).float()

	if negative_only == False:
		pos_sample_vector = torch.zeros(pos_num_vector.size()).float()
	else:
		pos_sample_vector = []	

	# compute accuracy
	rn = neg_num_matrix.size()[0]
	cn = neg_num_matrix.size()[1]
	for i in list(range(rn)):
		for j in list(range(cn)):
			if neg_num_matrix[i][j] > 0:
				assert neg_num_matrix[i][j] >= neg_correct_matrix[i][j]
				neg_sample_matrix[i][j] = neg_correct_matrix[i][j] / neg_num_matrix[i][j]

	if negative_only == False:
		ccn = pos_num_vector.size()[1]
		for i in list(range(ccn)):
			if pos_num_vector[0][i] > 0:
				assert pos_num_vector[0][i] >= pos_correct_vector[0][i]
				pos_sample_vector[0][i] = pos_correct_vector[0][i] / pos_num_vector[0][i]			
	#neg_sample_matrix = neg_correct_matrix / neg_num_matrix
	#pos_sample_vector = pos_correct_vector / pos_num_vector

	#print(neg_sample_matrix.sum())
	#print(pos_sample_vector.sum())

	# save accuracy
	neg_accuracy_matrix = neg_sample_matrix.clone()

	if negative_only == False:
		pos_accuracy_vector = pos_sample_vector.clone()
	else:
		pos_accuracy_vector = []	

	
	# convert accuracy to sampling probability (prob is 1+eps when this class is never seen before or accuracy=0)
	neg_sample_matrix = 1 - neg_sample_matrix + eps

	if negative_only == False:
		pos_sample_vector = 1 - pos_sample_vector + eps

	# replace NaN with eps (prob=eps for both class not chosen before or not classify well)
	#neg_sample_matrix[neg_sample_matrix != neg_sample_matrix] = eps
	#pos_sample_vector[pos_sample_vector != pos_sample_vector] = eps

	# ensure that diagnal of neg_num_matrix is 0
	d = min(neg_sample_matrix.size())
	for i in list(range(d)):
		neg_sample_matrix[i][i] = 0

	# ensure that the first row of neg_num_matrix is 0
	for i in list(range(cn)):
		neg_sample_matrix[0][i] = 0

	# normalization for the whole matrix
	#neg_sample_matrix = neg_sample_matrix / neg_sample_matrix.sum(1).view(-1,1).expand(neg_sample_matrix.size())
	if unbalanced == False:
		neg_sample_matrix = neg_sample_matrix / neg_sample_matrix.sum()
		# normalization along row
		if negative_only == False:
			pos_sample_vector = pos_sample_vector / pos_sample_vector.sum()
	else:
		prob_sum = neg_sample_matrix.sum() + pos_sample_vector.sum()
		neg_sample_matrix = neg_sample_matrix / prob_sum		
		pos_sample_vector = pos_sample_vector / prob_sum
	#print(pos_sample_vector)
	#print(N)

	# from prob to number
	if unbalanced == False:
		neg_sample_matrix = torch.round(neg_sample_matrix * N)
		if negative_only == False:
			pos_sample_vector = torch.round(pos_sample_vector * N)
	else:
		neg_sample_matrix = torch.round(neg_sample_matrix * N)
		pos_sample_vector = torch.round(pos_sample_vector * N)		

	# convert to int
	neg_sample_matrix = neg_sample_matrix.int()

	if negative_only == False:
		pos_sample_vector = pos_sample_vector.int()


	# check equality of negatives and positives
	if unbalanced == False:
		# for negative
		nn = neg_sample_matrix.sum()
		# reduce
		if nn > N:
			r = nn - N
			print("negative: reduce "+str(r))
			inds = random.sample(list(range(rn*cn)), r)
			for ind in inds:
				ri, ci = ind2rowcol(ind, cn)
				while True:
					if neg_sample_matrix[ri][ci] < 1:
						ind = random.sample(list(range(rn*cn)), 1)[0]
						ri, ci = ind2rowcol(ind, cn)
					else:
						break	

				neg_sample_matrix[ri][ci] -= 1
				assert neg_sample_matrix[ri][ci] >= 0
		# add		
		elif nn < N:
			r = N - nn
			print("negative: add "+str(r))
			inds = random.sample(list(range(rn*cn)), r)
			for ind in inds:
				ri, ci = ind2rowcol(ind, cn)
				while True:
					if ri == ci or ri == 0:
						ind = random.sample(list(range(rn*cn)), 1)[0]
						ri, ci = ind2rowcol(ind, cn)
					else:
						break	

				neg_sample_matrix[ri][ci] += 1
		
		# for positive
		if negative_only == False:
			pn = pos_sample_vector.sum()
			# reduce
			if pn > N:
				r = pn - N
				print("positive: reduce "+str(r))
				inds = random.sample(list(range(ccn)), r)
				for ind in inds:
					while True:
						if pos_sample_vector[0][ind] < 1:
							ind = random.sample(list(range(ccn)), 1)[0]
						else:
							break	
						
					pos_sample_vector[0][ind] -= 1
					assert pos_sample_vector[0][ind] >= 0
			# add		
			elif pn < N:
				r = N - pn
				print("positive: add "+str(r))
				inds = random.sample(list(range(ccn)), r)
				for ind in inds:
					pos_sample_vector[0][ind] += 1

	# check equality to N				
	else:	
		kn = neg_sample_matrix.sum() + pos_sample_vector.sum()
		#kn = N + 14729
		# reduce
		if kn > N:
			r = kn - N
			print("Reduce "+str(r)+" samples")
			# the matrix and vector are sparse
			ll = collect_positive_elements(neg_sample_matrix, pos_sample_vector)
			print("sampling ...")
			inds = random.sample(ll, r)
			for ind in inds:
				# negative
				if ind < rn*cn:
					ri, ci = ind2rowcol(ind, cn)
					neg_sample_matrix[ri][ci] -= 1
					assert neg_sample_matrix[ri][ci] >= 0
				# positive	
				else:
					ind = ind - rn*cn
					pos_sample_vector[0][ind] -= 1
					assert pos_sample_vector[0][ind] >= 0	
			'''
			assert rn*cn+ccn >= r
			inds = random.sample(list(range(rn*cn+ccn)), r)
			for ind in inds:
				# negative
				if ind < rn*cn:
					ri, ci = ind2rowcol(ind, cn)
					while True:
						if neg_sample_matrix[ri][ci] < 1:
							ind = random.sample(list(range(rn*cn)), 1)[0]
							ri, ci = ind2rowcol(ind, cn)
						else:
							break

					neg_sample_matrix[ri][ci] -= 1
					assert neg_sample_matrix[ri][ci] >= 0
				# positive	
				else:
					ind = ind - rn*cn
					while True:
						if pos_sample_vector[0][ind] < 1:
							ind = random.sample(list(range(ccn)), 1)[0]
						else:
							break	
						
					pos_sample_vector[0][ind] -= 1
					assert pos_sample_vector[0][ind] >= 0
			'''		
		# add		
		elif kn < N:
			r = N - kn
			print("Add "+str(r)+" samples")
			assert rn*cn+ccn >= r
			inds = random.sample(list(range(rn*cn+ccn)), r)
			for ind in inds:
				# negative
				if ind < rn*cn:
					ri, ci = ind2rowcol(ind, cn)
					while True:
						if ri == ci or ri == 0:
							ind = random.sample(list(range(rn*cn)), 1)[0]
							ri, ci = ind2rowcol(ind, cn)
						else:
							break	

					neg_sample_matrix[ri][ci] += 1
				# positive	
				else:
					ind = ind - rn*cn
					pos_sample_vector[0][ind] += 1
						

	# check correctness
	if unbalanced == False:
		nn = 0
		for i in list(range(rn)):
			for j in list(range(cn)):
				if i == 0:
					assert neg_sample_matrix[i][j] == 0
				if i == j:
					assert neg_sample_matrix[i][j] == 0

				assert neg_sample_matrix[i][j] >= 0
			
				nn += neg_sample_matrix[i][j]

		assert nn == N

		if negative_only == False:
			nn = 0
			for i in list(range(ccn)):
				assert pos_sample_vector[0][i] >= 0
				nn += pos_sample_vector[0][i]

			assert nn == N
	else:
		kk = 0
		for i in list(range(rn)):
			for j in list(range(cn)):
				if i == 0:
					assert neg_sample_matrix[i][j] == 0
				if i == j:
					assert neg_sample_matrix[i][j] == 0

				assert neg_sample_matrix[i][j] >= 0
			
				kk += neg_sample_matrix[i][j]


		for i in list(range(ccn)):
			assert pos_sample_vector[0][i] >= 0
			kk += pos_sample_vector[0][i]

		#print(str(kk))
		assert kk == N			
		
	#print(neg_sample_matrix.sum())	
	#print(pos_sample_vector.sum())	
	
	#print(pos_sample_vector)
	return neg_sample_matrix, pos_sample_vector, neg_accuracy_matrix, pos_accuracy_vector


def collect_positive_elements(neg_sample_matrix, pos_sample_vector):
	print("Collecting positive elements in negative matrix and positive vector ...")
	rn = neg_sample_matrix.size()[0]
	cn = neg_sample_matrix.size()[1]
	ccn = pos_sample_vector.size()[1]

	# allocate memory
	p = 0
	for i in list(range(rn)):
		for j in list(range(cn)):
			if neg_sample_matrix[i][j] > 0:
				p += neg_sample_matrix[i][j]

	for i in list(range(ccn)):
		if pos_sample_vector[0][i] >= 0:
			p += pos_sample_vector[0][i]

	l = [None] * p

	k = 0
	# save index
	for i in list(range(rn)):
		for j in list(range(cn)):
			if neg_sample_matrix[i][j] > 0:
				ind = rowcol2ind(i, j, cn)
				for g in list(range(k, k+neg_sample_matrix[i][j])):
					l[g] = ind

				k += neg_sample_matrix[i][j]

	for i in list(range(ccn)):
		if pos_sample_vector[0][i] >= 0:
			ind = i + rn*cn
			for g in list(range(k, k+pos_sample_vector[0][i])):
					l[g] = ind

			k += pos_sample_vector[0][i]

	#print(k)
	#print(len(l))			
	assert k == len(l)						

	return l


# read mapping between the image and the right clean logo 
def read_clean_logo_map(clean_logo_list):
	clean_logo_map = np.load(clean_logo_list).item()

	return clean_logo_map


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

# read logo patches
# format: patch_id class_id path
def read_logo_patches(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	print('Read logo patches: '+str(np.shape(lines)[0]))
	return lines[:,1], lines[:,2]
	

# for each class, gather real logo patches
# for class 0, gather real no-logo patches
def build_patch_bins(pos_file_name, neg_file_name):
	#patch_dir = os.path.join(dataset_path, 'patches')
	# construct logo bins, class_id is indexed from 1
	#pos_file_name = os.path.join(patch_dir, spec+'_pos_list.txt')
	print("Constructing logo bins from logo patch list: "+pos_file_name)
	with open(pos_file_name, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	lines = lines[:, 1:3]

	N = np.shape(lines)[0]
	
	bins = {}
	for line in lines:
		class_id = int(line[0])
		path = line[1]
		if class_id not in bins:
			bins[class_id] = [path]
		else:	
			bins[class_id].append(path)

	# check correctness		
	n = 0
	
	for k, v in bins.items():
		n += len(v)
	
	assert n == N
	print('Read logo patches: '+str(n))

	# add no-logo real patches to bin[0]
	print("Constructing no-logo bin from no-logo patch list: "+neg_file_name)
	with open(neg_file_name, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	bins[0] = lines[:,1].tolist()
	print('Read no logo patches: '+str(len(bins[0])))
	print('Bin size: '+str(len(bins)))

	# keys = []
	# for k, v in bins.items():
	# 	#print(str(k)+": "+str(len(v)))
	# 	keys.append(k)
	# #print(bins[25])
	# keys.sort()
	# print(keys)	

	return bins


# not include class 0 no_logo (include all classes, train and test)
def read_class_map_pos(class_list):
	class_map = np.load(class_list).item()
	#print("Loaded "+str(len(class_map))+" classes.")
	return class_map

# include class 0 no_logo (include all classes, train and test)
def read_class_map(class_list):	
	class_map = np.load(class_list).item()
	class_map[0] = "no_logo"

	return class_map

# map class id of training classes to index starting from 1
def read_train_class(train_class_file):	
	with open(train_class_file, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)

	# class id => index from 1 
	train_index_2_class_id = {}
	train_class_id_2_index = {}
	n = lines.shape[0]
	for i in list(range(n)):
		class_id = int(lines[i][0])
		train_class_id_2_index[class_id] = i + 1
		train_index_2_class_id[i+1] = class_id

	return train_class_id_2_index, train_index_2_class_id	

# dump a list of strings to a file
def dump(l, file):
	print("Dump to " + file)
	with open(file, 'w') as f:
		for line in l:
			f.write(line)
		
	print("Done.")


# get the number of clean logos in this brand
def get_logo_n(clean_logo_folder_path):
	i = 0
	for logo in os.listdir(clean_logo_folder_path):
		if logo.endswith(".jpg"):
			i += 1

	return i

def patch2cleanlogo(patch_path, clean_logo_map):
	# extract image name from patch_path
	last_occ = patch_path.rfind('_')
	image_path = patch_path[:last_occ]
	clean_logo_index = clean_logo_map[image_path]

	return clean_logo_index

# generate negative pairs according to the matrix
# format: source_image_path, target_image_path, label, source_class_id, target_class_id
def generate_negative_pairs(root_dir, patch_dir, clean_logo_path, class_map, neg_sample_matrix, bins, train_index_2_class_id):
	start_time = time.time()
	neg_pairs = []

	rn, cn = neg_sample_matrix.size()
	for r in list(range(rn)):
		target_class_id = matrix_index_to_class_id(r, train_index_2_class_id)
		target_class_name = class_map[target_class_id]
		# target class is not no-logo
		if r > 0:
			clean_logo_folder = os.path.join(clean_logo_path, target_class_name)
			# how many clean logos this brand has
			N = get_logo_n(os.path.join(root_dir, clean_logo_folder))

		for c in list(range(cn)):
			source_class_id = matrix_index_to_class_id(c, train_index_2_class_id)
			if neg_sample_matrix[r][c] > 0:
				# sample in this class
				patch_list = sample_one_bin(bins, source_class_id, neg_sample_matrix[r][c])
				# construct negative pairs
				for sample in patch_list:
					neg_patch_full_path = os.path.join(patch_dir, sample)
					clean_logo_full_path = os.path.join(clean_logo_folder, str(random.randint(0,N-1))+'.jpg')

					# check path correctness
					assert(os.path.exists(os.path.join(root_dir, neg_patch_full_path)))
					assert(os.path.exists(os.path.join(root_dir, clean_logo_full_path)))
					# add to final list
					neg_pairs.append((neg_patch_full_path, clean_logo_full_path, 0, source_class_id, target_class_id))

				assert neg_sample_matrix[r][c] == len(patch_list)
					
			elif neg_sample_matrix[r][c] < 0:	
				print("error: neg_sample_matrix element < 0")		

	assert len(neg_pairs) == neg_sample_matrix.sum()
	
	print("==> Generated "+str(len(neg_pairs))+" negative pairs.")
	print('==> Time: '+str(datetime.timedelta(seconds=time.time()-start_time)))
	return neg_pairs

# generate positive pairs
# format: source_image_path, target_image_path, label, source_class_id, target_class_id
# 1*296 
def generate_positive_pairs(root_dir, patch_dir, clean_logo_path, class_map, clean_logo_map, pos_sample_vector, bins, train_index_2_class_id):
	start_time = time.time()
	pos_pairs = []

	for c in list(range(1, pos_sample_vector.size()[1]+1)):
		class_id = matrix_index_to_class_id(c, train_index_2_class_id)
		class_name = class_map[class_id]
		clean_logo_folder = os.path.join(clean_logo_path, class_name)
		if pos_sample_vector[0][c-1] > 0:
			# sample in this class
			patch_list = sample_one_bin(bins, class_id, pos_sample_vector[0][c-1])
			for sample in patch_list:
				pos_patch_full_path = os.path.join(patch_dir, sample)
				clean_logo_full_path = os.path.join(clean_logo_folder, patch2cleanlogo(sample, clean_logo_map))

				# check path correctness
				assert(os.path.exists(os.path.join(root_dir, pos_patch_full_path)))
				assert(os.path.exists(os.path.join(root_dir, clean_logo_full_path)))
				# add to final list
				pos_pairs.append((pos_patch_full_path, clean_logo_full_path, 1, class_id, class_id))

	assert len(pos_pairs) == pos_sample_vector.sum()		

	print("==> Generated "+str(len(pos_pairs))+" positive pairs.")
	print('==> Time: '+str(datetime.timedelta(seconds=time.time()-start_time)))
	
	return pos_pairs

# main
if __name__ == '__main__':
	
	#root_dir = '/work/meng/data/logosc_300'
	
	class_list = os.path.join(dataset_path, 'split/class_map.npy')
	patch_dir = 'patches'
	pair_dir = os.path.join(dataset_path, 'pairs_dyn')
	clean_logo_path = 'clean_logos'
	clean_logo_png_path = 'clean_logos_png'
	
	train_class_file = os.path.join(os.path.join(dataset_path, "split"), "train_class.txt")

	#class_map = read_class_map(class_list)

	#bins = build_patch_bins(root_dir, 'train')

	matrix_path = "/work/meng/uvn/exps/siamese_bn_hm/finetune_all/matrix"
	
	#logo_rate = 0.5

	#if not os.path.exists(pair_dir): 
	#	os.makedirs(pair_dir)

	# redirect output 
	#sys.stdout = open(os.path.join(pair_dir, 'fact.txt'), 'w')

	# get clean logo map 
	#clean_logo_map = read_clean_logo_map(os.path.join(os.path.join(root_dir, clean_logo_png_path), 'clean_logo_map.npy'))
	'''
	# logo bins tell us for each class, what images do we have, should be modified dynamically
	logo_bins = build_logo_bins(os.path.join(os.path.join(root_dir, patch_dir), 'train_pos_list.txt'))
	fixed_logo_bins = build_logo_bins(os.path.join(os.path.join(root_dir, patch_dir), 'train_pos_list.txt'))
	# logo and no logo counts tell us for each class, how many logo negative pairs and no logo negative pairs do we need, should keep unchanged 
	logo_counts, no_logo_counts = split_logo_bins(fixed_logo_bins, logo_rate)
	# list of no logo patches
	no_logo_list = read_no_logo_patches(os.path.join(os.path.join(root_dir, patch_dir), 'train_neg_list.txt'))
	# map of class id and class name
	class_map = read_class_map_pos(class_list)
	# generate negative pairs repeatedly while training
	for i in range(10):
		print("==> Negative sampling, round "+str(i))
		train_neg_list, logo_bins, no_logo_list = generate_neg_pairs(class_list, root_dir, patch_dir, clean_logo_path, 
			logo_rate, clean_logo_map, logo_bins, no_logo_list, logo_counts, no_logo_counts, class_map, fixed_logo_bins, 'train')
		print("==> Sampled negative samples: "+str(len(train_neg_list)))
	
	print("Done.")

	# class_map_txt_to_npy(os.path.join(root_dir, 'split/class_map.txt'), class_list)
	# class_map = read_class_map_pos(class_list)
	# print(class_map)
	# print(str(len(class_map)))
	'''
	
	#train_class_id_2_index, train_index_2_class_id = read_train_class(train_class_file)
	# print("train_class_id_2_index")
	# print(len(train_class_id_2_index))
	# print("train_index_2_class_id")
	# print(train_index_2_class_id)
	
	# neg_sample_matrix_1 = torch.from_numpy(np.load(os.path.join(matrix_path, "neg_sample_matrix_1.npy")))
	# neg_sample_matrix_2 = torch.from_numpy(np.load(os.path.join(matrix_path, "neg_sample_matrix_2.npy")))
	# pos_sample_vector_2 = torch.from_numpy(np.load(os.path.join(matrix_path, "pos_sample_vector_2.npy")))
	# pos_sample_vector_1 = torch.from_numpy(np.load(os.path.join(matrix_path, "pos_sample_vector_1.npy")))

	#print(neg_sample_matrix_1.sum())
	#print(neg_sample_matrix_2.sum())
	#print(pos_sample_vector_1.sum())
	#print(pos_sample_vector_2.sum())

	#generate_positive_pairs(root_dir, patch_dir, clean_logo_path, class_map, clean_logo_map, pos_sample_vector_2, bins, train_index_2_class_id)
	#generate_negative_pairs(root_dir, patch_dir, clean_logo_path, class_map, neg_sample_matrix_2, bins, train_index_2_class_id)
	#test_matrix(train_class_id_2_index)
	