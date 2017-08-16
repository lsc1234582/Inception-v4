import pickle
import os
import numpy as np
import matplotlib.image as mpimg
import pprint
from PIL import Image
import pickle
import random

IMAGE_DIMS = (299, 299)
# First 500 images
OUT_FILE = 'dataset_batch_01'

def get_image_file_path(root):
    for path, dirs, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[-1] == '.jpg':
                yield os.path.join(path, f)
        for d in dirs:
            get_image_file_path(d)

##### UI #####
progress_bar = "|" + " " * 100 + "|"

labels_index_map = {}
dataset = {}

label_index = 0
labels = []
images = []
img_files = list(get_image_file_path('.'))
random.shuffle(img_files)
#print(len(img_files))

for img_file in img_files[:500]:
    img = Image.open(img_file)
    img = img.convert(mode='RGB')
    img_array = np.array(img.resize(IMAGE_DIMS)).reshape(1, -1)
    label = img_file.split('/')[-2]
    index = labels_index_map.get(label, -1)
    if index == -1:
        labels_index_map[label] = label_index
        label_index += 1
    images.append(img_array)
    index = labels_index_map[label]
    labels.append(index)

dataset['data'] = np.concatenate(images, axis=0)
dataset['labels'] = np.array(labels)

with open(OUT_FILE, 'wb') as f:
    pic = pickle.Pickler(f)
    pic.dump(dataset)

pprint.pprint(dataset['data'].shape)
