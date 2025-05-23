import os
import sys
import math
import random
import shutil
import time
import numpy as np
import scipy.misc
import xmltodict
from PIL import Image
import json
import urllib.request

sys.path.append("../")
from config import *

def get_clean_logo_path(image_path):
	jason_file = os.path.splitext(image_path)[0] + '.json'
	with open(jason_file) as json_data:
		d = json.load(json_data)
		return d["detections_iconUrl"]

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


# map each image to the local location of correct clean logo
def iterate_all_images(image_dir, clean_logo_png_dir):

	print('Start collecting paths for the whole dataset...')	
	clean_logo_map = {}
	with open(os.path.join(clean_logo_png_dir, 'clean_logo_map.txt'), 'w') as f: 
		i = 0
		for folder in os.listdir(image_dir):
			folder_full = os.path.join(image_dir, folder)
			if os.path.isdir(folder_full):
				urls = set()
				# for all images in this brand, collect urls
				for file in os.listdir(folder_full):
					if file.endswith('.jpg'):
						image_path = os.path.join(folder_full, file)
						url = get_clean_logo_path(image_path)
						urls.add(url)
				# brand name: # of logos
				print(folder+": "+str(len(urls)))		
				# download clean logos for this brand as .png files, index them from 0
				brand_dir = os.path.join(clean_logo_png_dir, folder)
				if not os.path.exists(brand_dir): 
					os.makedirs(brand_dir)
				else:
					clear_dir(brand_dir)
				# dump
				url_map = {}
				j = 0
				for url in urls:
					urllib.request.urlretrieve(url, os.path.join(brand_dir, str(j)+".png"))
					url_map[url] = j
					j += 1

				assert (j == len(urls))	
				
				# for each image in this brand, establish mapping
				for file in os.listdir(folder_full):
					if file.endswith('.jpg'):
						image_path = os.path.join(folder_full, file)
						url = get_clean_logo_path(image_path)
						index = url_map[url]
						clean_logo_map[os.path.join(folder, os.path.splitext(file)[0])] = str(index)+".jpg"
						f.write(os.path.join(folder, os.path.splitext(file)[0])+" "+str(index)+".jpg\n")
					
				# next brand		
				i += 1
	# saving the mapping		
	np.save(os.path.join(clean_logo_png_dir, 'clean_logo_map.npy'), clean_logo_map)
	print('%d brands Completed!' % (i))

# main function 
if __name__ == '__main__':

	clean_logo_png_dir = os.path.join(dataset_path, 'clean_logos_png')

	# build or clean the clean_logo_png_dir
	if not os.path.exists(clean_logo_png_dir): 
		os.makedirs(clean_logo_png_dir)
	else:
		clear_dir(clean_logo_png_dir)

	iterate_all_images(image_dir, clean_logo_png_dir)


