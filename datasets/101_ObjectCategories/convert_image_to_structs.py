import matplotlib.image as mpimg
import numpy as np
import os
import pickle
import pprint
from PIL import Image
import random
import sys

# Number of images contained in a single batch
BATCH_SIZE=1000
IMAGE_DIMS = (299, 299)
OUT_FILE_BASE = 'dataset_batch'
INDEX_LABEL_MAP_FILE = 'label_names'

def get_image_file_path(root):
    for path, dirs, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[-1] == '.jpg':
                yield os.path.join(path, f)
        for d in dirs:
            get_image_file_path(d)

def process_and_store_batch(batch_indx, img_indx_start, img_indx_end, img_files, labels_index_map, label_index,
        label_names, struct_dir):
    dataset = {}
    labels = []
    images = []
    for img_file in img_files[img_indx_start : img_indx_end]:
        img = Image.open(img_file)
        img = img.convert(mode='RGB')
        img_array = np.array(img.resize(IMAGE_DIMS)).reshape(1, -1)
        label = img_file.split('/')[-2]
        index = labels_index_map.get(label, -1)
        # If label does not exist yet
        if index == -1:
            labels_index_map[label] = label_index
            label_names.append(label)
            label_index += 1
        images.append(img_array)
        index = labels_index_map[label]
        labels.append(index)
    out_file = OUT_FILE_BASE + '_' + str(batch_indx)
    dataset['data'] = np.concatenate(images, axis=0)
    dataset['labels'] = np.array(labels)

    with open(os.path.join(struct_dir, out_file), 'wb') as f:
        pic = pickle.Pickler(f)
        pic.dump(dataset)

##### UI #####
progress_bar = "|" + " " * 100 + "|"

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: python convert_image_to_structs.py image_dir struct_dir")
	# Auxiliary map to help keep track of labels
	labels_index_map = {}
	# Index to label map that is actually stored
	label_names = []
	label_index = 0

	img_files = list(get_image_file_path(sys.argv[1]))
	struct_dir = sys.argv[2]
	random.shuffle(img_files)
	total_num_images = len(img_files)

	batch_indx = 0
	# Process and store batches of multiples of BATCH_SIZE number of images
	while batch_indx < (total_num_images // BATCH_SIZE):
	    print("Processing and storing batch " + str(batch_indx))
	    process_and_store_batch(batch_indx, batch_indx * BATCH_SIZE, (batch_indx + 1) * BATCH_SIZE, img_files,
		    labels_index_map, label_index, label_names, struct_dir)
	    batch_indx += 1
	# Process and store the renaming images in the last batch
	print("Processing and storing batch " + str(batch_indx))
	process_and_store_batch(batch_indx, batch_indx * BATCH_SIZE, total_num_images, img_files, labels_index_map,
		label_index, label_names, struct_dir)

	# Store index to label mapping
	with open(os.path.join(struct_dir, INDEX_LABEL_MAP_FILE), 'wb') as f:
	    pic = pickle.Pickler(f)
	    pic.dump(label_names)

	print("Finished processing batches of images!")
