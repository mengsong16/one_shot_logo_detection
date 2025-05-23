import os
import sys
import math
import random
import shutil
import time
import numpy as np
from difflib import SequenceMatcher
import Levenshtein
from PIL import Image
import matplotlib.pyplot as plt

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

def similar(t, s):
	return SequenceMatcher(None, t, s).ratio()

def find_most_similar_string(target_str, source_strs):
	print(target_str)
	res = ""

	max_sim = -np.inf
	for source_str in source_strs:
		# covert to lower case
		score = Levenshtein.ratio(str.lower(target_str), str.lower(source_str))
		if score > max_sim:
			max_sim = score
			res = source_str

	return res		
	

def copy_clean_logos(raw_logo_dir, output_dir, class_list):
	# numpy load a dictionary
	classes = np.load(class_list).item()
	# collect target class names
	target_logos = []
	for k, v in classes.items():
		target_logos.append(v)

	print('Target classes: ', len(target_logos))
	
	# collect raw class names
	raw_logos = []
	for folder in os.listdir(raw_logo_dir):
		folder_full = os.path.join(raw_logo_dir, folder)
		if os.path.isdir(folder_full):
			raw_logos.append(folder)

	raw_logos.sort()		
	print('Total logos: ', len(raw_logos))
	
	print('Matched pairs:')
	# match target to raw
	matched_map = {}
	for t in target_logos:
		matched_map[t] = find_most_similar_string(t, raw_logos)
	
	for k, v in matched_map.items():
		print(k + ': '+ v)
	
	# output clean logos, name it as target	
	# if multipul clean logos exist, use the first one ended with .png
	print("Copy images ...")
	for k, v in matched_map.items():
		# source path
		source_folder = os.path.join(raw_logo_dir, v)
		target_folder = os.path.join(output_dir_full, k)
		if not os.path.exists(target_folder): 
			os.makedirs(target_folder)
		else:	
			clear_dir(target_folder)
		# copy all files in source_folder to target_folder
		for file in os.listdir(source_folder):
			if file.endswith(".png"):
				source_file = os.path.join(source_folder, file)
				target_file = os.path.join(target_folder, file)
				shutil.copy2(source_file, target_file)
		
	

# main
if __name__ == '__main__':

	root_dir = '/work/meng/data/logosc_300'
	raw_logo_dir = os.path.join(root_dir, 'logo_icons')
	output_dir_full = os.path.join(root_dir, 'clean_logos_png')
	class_list = os.path.join(root_dir, 'split/class_map.npy')


	if not os.path.exists(output_dir_full): 
		os.makedirs(output_dir_full)
	else:	
		clear_dir(output_dir_full)	

	print('Start copying clean logos...')
	copy_clean_logos(raw_logo_dir, output_dir_full, class_list)


	print('Done.')