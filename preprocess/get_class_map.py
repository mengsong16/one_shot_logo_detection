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


# collect brand information
def collect_brand_id_name(csv_file):
	df = pd.read_csv(csv_file, usecols=['detections_name', 'detections_id'])
	#print(len(df))
	
	res = {}
	for i in list(range(len(df))):
		l = df.iloc[i,:]
		brand_id = int(l['detections_id'])
		brand_name = str(l['detections_name'])
		#print(brand_name)
		
		if brand_id not in res:
			res[brand_id] = brand_name
		else:
			if res[brand_id] != brand_name:
				print("Inconsistent Error: "+str(brand_id)+": "+res[brand_id]+", "+brand_name)	
		
	print("Dictionary: "+str(len(res)))
	print("Done: "+csv_file+", "+str(len(df)))

	return res
	
# save class mapping
def save_class_mapping(class_map, class_map_dir):
	# sort by id
	class_map = OrderedDict(sorted(class_map.items()))
	# saving class mapping		
	with open(os.path.join(class_map_dir, 'class_map.txt'), 'w') as f:
		for key, value in class_map.items():
			f.write(str(key)+" "+str(value)+"\n")

	
	np.save(os.path.join(class_map_dir, 'class_map.npy'), class_map)

# merge dictionary b to a
def merge(a, b):
	print("Before merging:"+str(len(a))+", "+str(len(b)))
	for key, value in b.items():
		if key not in a:
			a[key] = value
		else:
			if a[key] != value:
				print("Inconsistent Error: "+str(key)+": "+a[key]+", "+value)

	print("After merging:"+str(len(a)))
	return a
			
# main function 
if __name__ == '__main__':
	root_dir = '/Users/mengsong/Desktop/logo-detection'

	# redirect output 
	sys.stdout = open(os.path.join(root_dir, 'fact_class_map.txt'), 'w')

	print("Collecting...")
	train_csv = collect_brand_id_name(os.path.join(root_dir, 'train_flickr100m.csv'))
	test_wo32_csv = collect_brand_id_name(os.path.join(root_dir, 'test_unseen_flickr100m.csv'))
	test_w32_csv = collect_brand_id_name(os.path.join(root_dir, 'test_unseen_flickr32_flickr100m.csv'))
	test_train_class_csv = collect_brand_id_name(os.path.join(root_dir, 'test_flickr100m_seen.csv'))
	test_bkgr_csv = collect_brand_id_name(os.path.join(root_dir, 'test_background_flickr100m.csv'))
	val_csv = collect_brand_id_name(os.path.join(root_dir, 'val_unseen_flickr100m.csv'))
	val_train_class_csv = collect_brand_id_name(os.path.join(root_dir, 'val_flickr100m.csv'))
	val_bkgr_csv = collect_brand_id_name(os.path.join(root_dir, 'val_background_flickr100m.csv'))

	print("Merging...")
	dic = merge(train_csv, test_wo32_csv)
	dic = merge(dic, test_w32_csv)
	dic = merge(dic, test_train_class_csv)
	dic = merge(dic, test_bkgr_csv)
	dic = merge(dic, val_csv)
	dic = merge(dic, val_train_class_csv)
	dic = merge(dic, val_bkgr_csv)

	print("Saving...")
	save_class_mapping(dic, root_dir)
	print("Done.")
	

	

