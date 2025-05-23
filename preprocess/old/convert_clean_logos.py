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

sys.path.append("../")
from config import *

white_names = {"Aquafina/2.png", "Boston_Celtics/1.png", "Bud_Light/0.png", "Bud_Light/5.png", "Budweiser/0.png", "Budweiser/1.png", 
	"Chick-fil-a/2.png", "Coca-Cola/3.png", "Colgate/0.png", "Dannon/2.png", "Dannon/3.png", "Dodge/0.png", 
	"Facebook/0.png", "Facebook/1.png", "Fiji/0.png", "FOX_News/0.png", "Fox_Racing/0.png", "Gatorade/3.png", "Heineken/3.png", 
	"Home_Depot/0.png", "IN-N-OUT_BURGER/2.png", "Instagram/0.png", "Jack_in_the_box/0.png", "Johnny_Rockets/0.png", "KFC/1.png", 
	"Kroger/0.png", "LA_Clippers/1.png", "Land_Rover/0.png", "Lays/2.png", "Lenovo/0.png", "Levis/2.png", "Liverpool_FC/0.png", 
	"Marvel/0.png", "Marvel/1.png", "McDonalds/1.png", "Microsoft/1.png", "Milka/0.png", "Nescafe/1.png", "Pabst_Blue_Ribbon_Beer/0.png", 
	"Panera_Bread/1.png", "Pepsi/0.png", "Pepsi/3.png", "Perrier/1.png", "Pizza_Hut/5.png", "Play_Station/2.png", "Poland_Spring/0.png", 
	"Poland_Spring/1.png", "Redds_Apple_Ale/0.png", "Redds_Apple_Ale/1.png", "Samsung/1.png", "Samuel_Adams/0.png", "Skittles/0.png", 
	"Smart_Water/0.png", "Smart_Water/1.png", "Smirnoff/0.png", "Smirnoff/1.png", "Smirnoff/2.png", "Smirnoff/3.png", "Star_Wars/0.png", 
	"Star_Wars/2.png", "Subway/1.png", "Taco_Bell/1.png", "Taco_Bell/3.png", "Uber/1.png", "Volcom/0.png", "Volkswagen/1.png", "Walgreens/2.png", 
	"Walkers/1.png", "Washington_Nationals/0.png", "Wendys/0.png"}

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

def is_white(logo_name):
	for name in white_names:
		if name == logo_name:
			return True

	return False

def png2jpg(logo_name, source_file, target_file):
	w = False
	with open(source_file, 'rb') as f:
		with Image.open(f) as img:
			#img.load() # required for png.split()
			#foreground = img.split()[0:3]
			if is_white(logo_name):
				# black background
				background =  Image.new('RGB', img.size)
				w = True
			else:	
				# white background
				background =  Image.new('RGB', img.size, 'white')

			img = Image.composite(img, background, img) # image1, image2, mask(could be RGBA, L ,1)
			img = img.convert('RGB')
			img.save(target_file, "JPEG")

	return w		

def convert_clean_logos(png_logo_dir, jpg_logo_dir):
	# convert png to jpg
	i = 0
	j = 0
	wn = 0
	for folder in os.listdir(png_logo_dir):
		# exclude fact.txt
		if os.path.isdir(os.path.join(png_logo_dir, folder)):
			source_folder = os.path.join(png_logo_dir, folder)
			target_folder = os.path.join(jpg_logo_dir, folder)
			if not os.path.exists(target_folder): 
				os.makedirs(target_folder)
			else:	
				clear_dir(target_folder)

			# convert all files in source_folder from png to jpg
			# and save in target_folder
			for file in os.listdir(source_folder):
				if file.endswith(".png"):
					source_file = os.path.join(source_folder, file)
					target_file = os.path.join(target_folder, os.path.splitext(file)[0] + '.jpg')
					if png2jpg(os.path.join(folder, file), source_file, target_file):
						wn += 1

					j += 1
			
			i += 1
	
	print(wn)
	print(len(white_names))
	#assert(wn == len(white_names))		

	print("Processed " + str(i) + " folders.")
	print("Converted " + str(j) + " logos.")
	print("Including white logos: " + str(wn))		

	 

# main
if __name__ == '__main__':

	png_logo_dir = os.path.join(dataset_path, 'clean_logos_png')
	jpg_logo_dir = os.path.join(dataset_path, 'clean_logos')
	
	if not os.path.exists(jpg_logo_dir): 
		os.makedirs(jpg_logo_dir)
	else:	
		clear_dir(jpg_logo_dir)

	# redirect output 
	sys.stdout = open(os.path.join(png_logo_dir, 'fact_convert.txt'), 'w')

	print('Start converting clean logos...')
	convert_clean_logos(png_logo_dir, jpg_logo_dir)
	print('Done.')