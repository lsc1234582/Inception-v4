#from keras.utils import plot_model
#from inception_v4 import create_inception_v4
import numpy as np
import pprint
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FILE = 'datasets/cifar-10-batches-py/data_batch_1'

def load_data(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def choose_random_image_data(data, num_images=1):
    images = [data[indx] for indx in (random.randrange(0, data.shape[0]) for _ in range(num_images))]
    return images

def display_images(images, subplot_layout):
    for image in images:
        plt.imshow(image.reshape((32, 32, 3)))


if __name__ == "__main__":
    #inception_v4 = create_inception_v4(load_weights=False)
    #inception_v4.compile('sgd', 'categorical_crossentropy', )
    #inception_v4.summary()

    #plot_model(inception_v4, to_file="Inception-v4.png", show_shapes=True)
    dataset = load_data(FILE)
    images = choose_random_image_data(dataset[b'data'], 10)
    #display_images(images, (1,1))
    plt.plot([1,2,3,4])
    plt.show()
    pprint.pprint(images)
