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


def read_bbox_xml(image_path, image_dir):
	image = Image.open(os.path.join(image_dir,image_path))
	width, height = image.size
	xml_file = os.path.splitext(image_path)[0] + '.xml'
	xml_path = os.path.join(image_dir, xml_file)
	if os.path.isfile(xml_path):
		data = xmltodict.parse(open(xml_path).read())
		objs = data['annotation']['object']
		if type(objs) is list:
			# assume single bbox (Could work for logosc300)
			obj = objs[0]
			bbox = obj['bndbox']
			xmin = int(bbox['xmin'])
			ymin = int(bbox['ymin'])
			xmax = int(bbox['xmax'])
			ymax = int(bbox['ymax'])
		else:
			bbox = objs['bndbox']
			xmin = int(bbox['xmin'])
			ymin = int(bbox['ymin'])
			xmax = int(bbox['xmax'])
			ymax = int(bbox['ymax'])

		p_width = xmax - xmin
		p_height = ymax - ymin	

		return p_width, p_height, width, height
	else:	
		print("Error: "+xml_path+" DOES NOT exist.")
		return 0

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

def read_image_list(image_list):
	with open(image_list, 'r') as f:
		lines = f.readlines()

	lines = [x.strip().split() for x in lines]
	lines = np.asarray(lines)
	print('Read images: '+str(np.shape(lines)[0]))

	return lines[:,2]


# compute min bbox
def find_min_bbox(image_dir, train_list, test_list, bbox_file):

	# read training and test images
	training_paths = read_image_list(train_list)
	test_paths = read_image_list(test_list)
	
	image_paths = training_paths.tolist() + test_paths.tolist()

	min_size = math.inf
	min_percentage = 1
	avg_width = 0
	avg_height = 0
	i = 0 
	for image_path in image_paths:
		p_width, p_height, width, height = read_bbox_xml(image_path, image_dir)
		p_min_size = min(p_width, p_height)
		p_min_percentage = min(p_width/float(width), p_height/float(height))
		if min_size > p_min_size:
			min_size = p_min_size
		if min_percentage > p_min_percentage:
			min_percentage = p_min_percentage

		avg_width += p_width
		avg_height += p_height	

		i += 1	

		'''
		if i % 500 == 0:
			print(str(i)+ ' Done.')	
		'''	

	avg_width /= float(i)
	avg_height /= float(i)		

	print("Min bbox size: " + str(min_size) +  " pixels")
	print("Min bbox percentage: " + str(min_percentage))	
	print("Avg bbox width: " + str(avg_width) +  " pixels")
	print("Avg bbox height: " + str(avg_height) +  " pixels")	
	# write results
	with open(os.path.join(bbox_file), 'w') as f:
		f.write(str(min_size) +'\n')
		f.write(str(p_min_percentage)+'\n')
		f.write(str(avg_width) +'\n')
		f.write(str(avg_height) +'\n')
	

	return

# main function 
if __name__ == '__main__':

	root_dir = '/work/meng/data/logosc_300'	
	image_dir = os.path.join(root_dir, 'logosc300')
	train_list = os.path.join(root_dir, 'split/train.txt')
	test_list = os.path.join(root_dir, 'split/test.txt')
	output_dir = os.path.join(root_dir, 'split')
	bbox_file = os.path.join(output_dir, 'min_bbox.txt')


	# redirect output 
	sys.stdout = open(os.path.join(output_dir, 'fact_bbox.txt'), 'w')

	find_min_bbox(image_dir, train_list, test_list, bbox_file)

	print('Done.')

