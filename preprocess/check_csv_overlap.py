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



# collect image name
def collect_brand_id_name(csv_file):
	df = pd.read_csv(csv_file, usecols=['image_base_name'])
	
	
	s = set()
	for i in list(range(len(df))):
		l = df.iloc[i,:]
		image_name = str(l['image_base_name'])
		#if image_name in s:
		#	print("Error: "+ image_name + " appear multiple times!")
		s.add(image_name)	
		
	print(csv_file+": "+str(len(s)))

	return s
	
# check overlap
def check_overlap(csv1, csv2):
	s1 = collect_brand_id_name(csv1)
	s2 = collect_brand_id_name(csv2)
	intersect = list(s1 & s2)
	
	print(str(len(intersect)) + " overlap between "+csv1 + ", "+csv2)
	for e in intersect:
		print(e)	

	print("======================================================")	



# main function 
if __name__ == '__main__':

	# redirect output 
	sys.stdout = open(os.path.join(fact_dir, 'fact_overlap.txt'), 'w')

	check_overlap(val_csv, val_seen_csv)
	check_overlap(test_wo32_csv, test_seen_csv)
	check_overlap(test_w32_csv, test_seen_csv)
	check_overlap(test_w32_csv, test_wo32_csv)
	

	

