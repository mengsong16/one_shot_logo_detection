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


# the integer is the max index (from 0)
def count_clean_logo_instances(csv_file, clean_logo_folder, spec):
	dic = {}

	df = pd.read_csv(csv_file, usecols=['brand_instance_logo', 'detections_id'])
	
	for i in list(range(len(df))):
		l = df.iloc[i,:]
		brand_id = int(l['detections_id'])
		# not nan
		if l['brand_instance_logo'] == l['brand_instance_logo']:
			clean_logo_instance = os.path.splitext(str(l['brand_instance_logo']))[0]
			ind = clean_logo_instance.rfind('_')
			num = int(clean_logo_instance[ind+1:])
			if brand_id not in dic:
				dic[brand_id] = set()

			dic[brand_id].add(num)
			
	
	# sort by id
	dic = OrderedDict(sorted(dic.items()))
	# save			
	save_counter(dic, clean_logo_folder, spec)	
	
	print("Done: "+csv_file+", dictionary size: "+str(len(dic)))
	print("=============================================================")
	
# save the counter
def save_counter(dic, clean_logo_dir, spec):	
	with open(os.path.join(clean_logo_dir, spec+'_clean_logo_map.txt'), 'w') as f:
		for key, value in dic.items():
			s = ','.join(str(e) for e in value)
			f.write(str(key)+" "+s+"\n")

	
	np.save(os.path.join(clean_logo_dir, spec+'_clean_logo_map.npy'), dic)


# main function 
if __name__ == '__main__':
	root_dir = '/Users/mengsong/Desktop/logo-detection'

	# redirect output 
	sys.stdout = open(os.path.join(fact_dir, 'fact_clean_logo_map.txt'), 'w')

	count_clean_logo_instances(train_csv, train_clean_logo_dir, 'train')
	count_clean_logo_instances(test_wo32_csv, test_clean_logo_dir, 'test_wo32')
	count_clean_logo_instances(test_w32_csv, test_clean_logo_dir, 'test_w32')
	count_clean_logo_instances(test_seen_csv, test_clean_logo_dir, 'test_seen')
	count_clean_logo_instances(val_csv, val_clean_logo_dir, 'val')
	count_clean_logo_instances(val_seen_csv, val_clean_logo_dir, 'val_seen')
	
	

	

