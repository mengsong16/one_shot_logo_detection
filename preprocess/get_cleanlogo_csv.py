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

sys.path.append("../")
from config import *

# get the class id in this csv file
def get_cleanlogos_in_csv(dataset_path, csv_file, output_path, output_file):
	df = pd.read_csv(os.path.join(dataset_path, csv_file), usecols=['detections_id', 'brand_instance_logo'])
	res = []
	
	s = set()
	
	for i in list(range(len(df))):
		l = df.iloc[i,:]
		brand_id = str(l['detections_id'])
		brand_instance = os.path.splitext(str(l['brand_instance_logo']))[0] + '.jpg'
		s.add((brand_id, brand_instance))

	s = sorted(s)	

	with open(os.path.join(output_path, output_file), 'w') as f:	
		for e in s:
			f.write(e[0]+','+e[1]+'\n')	
	
	print("Done: "+csv_file+", number of clean logo instances: "+str(len(s)))


# main function 
if __name__ == '__main__':

	# redirect output 
	sys.stdout = open(os.path.join(fact_dir, 'fact_cleanlogo_csv.txt'), 'w')

	print("Get clean logo instances for each csv file...")
	
	get_cleanlogos_in_csv(train_dir, 'train_flickr100m.csv', csv_cleanlogo_dir, 'train_cleanlogos.txt')
	get_cleanlogos_in_csv(test_dir, 'test_unseen_flickr100m.csv', csv_cleanlogo_dir, 'test_wo32_cleanlogos.txt')
	get_cleanlogos_in_csv(test_dir, 'test_unseen_flickr32_flickr100m.csv', csv_cleanlogo_dir, 'test_w32_cleanlogos.txt')
	get_cleanlogos_in_csv(test_dir, 'test_flickr100m_seen.csv', csv_cleanlogo_dir, 'test_seen_cleanlogos.txt')
	get_cleanlogos_in_csv(val_dir, 'val_unseen_flickr100m.csv', csv_cleanlogo_dir, 'val_cleanlogos.txt')
	get_cleanlogos_in_csv(val_dir, 'val_flickr100m.csv', csv_cleanlogo_dir, 'val_seen_cleanlogos.txt')

	

