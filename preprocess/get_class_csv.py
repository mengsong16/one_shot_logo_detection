import os
import sys
import math
import random
import shutil
import time
import numpy as np
import scipy.misc
from PIL import Image
import pandas as pd
from collections import OrderedDict


# get the class id in this csv file
def get_classes(dataset_path, csv_file, output_file):
	df = pd.read_csv(os.path.join(dataset_path, csv_file), usecols=['detections_id'])
	s = set()
	
	for i in list(range(len(df))):
		l = df.iloc[i,:]
		brand_id = int(l['detections_id'])
		s.add(brand_id)
			
	s = sorted(s)		

	with open(os.path.join(dataset_path, output_file), 'w') as f:
		for e in s:
			f.write(str(e)+"\n")
	
	print("Done: "+csv_file+", number of classes: "+str(len(s)))


# main function 
if __name__ == '__main__':
	root_dir = '/Users/mengsong/Desktop/logo-detection'

	# redirect output 
	sys.stdout = open(os.path.join(root_dir, 'fact_class_csv.txt'), 'w')

	print("Get classes for each csv file...")
	
	get_classes(root_dir, 'train_flickr100m.csv', 'train_class.txt')
	get_classes(root_dir, 'test_unseen_flickr100m.csv', 'test_wo32_class.txt')
	get_classes(root_dir, 'test_unseen_flickr32_flickr100m.csv', 'test_w32_class.txt')
	get_classes(root_dir, 'test_flickr100m_seen.csv', 'test_seen_class.txt')
	get_classes(root_dir, 'val_unseen_flickr100m.csv', 'val_class.txt')
	get_classes(root_dir, 'val_flickr100m.csv', 'val_seen_class.txt')

	

