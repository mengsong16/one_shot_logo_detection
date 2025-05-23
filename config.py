import os

# root path
root_dir = "/data"
# experiment path: include folders of variant experimental settings (Please ensure that this folder exist)
work_dir = os.path.join(root_dir, "uvn")
if not os.path.exists(work_dir): 
	os.makedirs(work_dir)
# dataset path: include original image dataset split into train/test/val
dataset_path = os.path.join(root_dir, "flickr_100m_logo_dataset/flickr_100m_logo")
# fact dir
fact_dir = os.path.join(dataset_path, 'fact')
if not os.path.exists(fact_dir): 
	os.makedirs(fact_dir)
# class map for all
class_map_file = os.path.join(dataset_path, 'class_map.npy')
# csv class folder
csv_class_dir = os.path.join(dataset_path, 'csv_class')	
# csv class folder
csv_cleanlogo_dir = os.path.join(dataset_path, 'csv_cleanlogo')
# train
train_dir = os.path.join(dataset_path, "train")
train_image_dir = os.path.join(train_dir, "images")
train_patch_dir = os.path.join(train_dir, "patches")
train_pair_dir = os.path.join(train_dir, "pairs")
train_clean_logo_dir = os.path.join(train_dir, "clean_logos")
train_csv = os.path.join(train_dir, "train_flickr100m.csv")
mean_file = os.path.join(train_dir, "mean.txt")

train_lmdb_dir = os.path.join(train_dir, "lmdb")
train_lmdb_file = os.path.join(train_lmdb_dir, "train.lmdb")

# test
test_dir = os.path.join(dataset_path, "test")
test_image_dir = os.path.join(test_dir, "images")
test_patch_dir = os.path.join(test_dir, "patches")
test_pair_dir = os.path.join(test_dir, "pairs")
test_clean_logo_dir = os.path.join(test_dir, "clean_logos")
test_wo32_csv = os.path.join(test_dir, "test_unseen_flickr100m.csv")
test_w32_csv = os.path.join(test_dir, "test_unseen_flickr32_flickr100m.csv")
test_seen_csv = os.path.join(test_dir, "test_flickr100m_seen.csv")
test_bkgr_csv = os.path.join(test_dir, "test_background_flickr100m.csv")

test_lmdb_dir = os.path.join(test_dir, "lmdb")
test_wo32_lmdb_file = os.path.join(test_lmdb_dir, "test_wo32.lmdb")
test_w32_lmdb_file = os.path.join(test_lmdb_dir, "test_w32.lmdb")
test_seen_lmdb_file = os.path.join(test_lmdb_dir, "test_seen.lmdb")

# validation
val_dir = os.path.join(dataset_path, "val")
val_image_dir = os.path.join(val_dir, "images")
val_patch_dir = os.path.join(val_dir, "patches")
val_pair_dir = os.path.join(val_dir, "pairs")
val_clean_logo_dir = os.path.join(val_dir, "clean_logos")
val_csv = os.path.join(val_dir, "val_unseen_flickr100m.csv")
val_seen_csv = os.path.join(val_dir, "val_flickr100m.csv")
val_bkgr_csv = os.path.join(val_dir, "val_background_flickr100m.csv")

val_lmdb_dir = os.path.join(val_dir, "lmdb")
val_lmdb_file = os.path.join(val_lmdb_dir, "val.lmdb")
val_seen_lmdb_file = os.path.join(val_lmdb_dir, "val_seen.lmdb")

