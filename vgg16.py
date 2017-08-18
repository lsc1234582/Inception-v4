from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K
from keras.utils.data_utils import get_file

def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(x)
    x = Activation('relu')(x)
    return x

def create_vgg16(nb_classes=1001, load_weights=True):
    if K.image_dim_ordering() == 'th':
        init = Input((3, 224, 224))
    else:
        init = Input((224, 224, 3))

    # Input Shape is 224 x 224 x 3 (tf) or 3 x 224 x 224 (th)
    x1 = conv_block(init, 64, 3, 3)
    x1 = conv_block(x1, 64, 3, 3)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='valid')(x1)
    
    x2 = conv_block(x2, 128, 3, 3)
    x2 = conv_block(x2, 128, 3, 3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='valid')(x2)

    x3 = conv_block(x3, 256, 3, 3)
    x3 = conv_block(x3, 256, 3, 3)
    x3 = conv_block(x3, 256, 3, 3)
    x4 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='valid')(x3)

    x4 = conv_block(x4, 512, 3, 3)
    x4 = conv_block(x4, 512, 3, 3)
    x4 = conv_block(x4, 512, 3, 3)
    x5 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='valid')(x4)

    x5 = conv_block(x5, 512, 3, 3)
    x5 = conv_block(x5, 512, 3, 3)
    x5 = conv_block(x5, 512, 3, 3)
    x6 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='valid')(x5)

    # Fully connected layers
    x6 = Flatten()(x6)
    x7 = Dense(output_dim=4096, activation='relu')(x6)
    x7 = Dropout(0.5)(x7)

    x8 = Dense(output_dim=4096, activation='relu')(x7)
    x8 = Dropout(0.5)(x8)

    # Output
    out = Dense(output_dim=nb_classes, activation='softmax')(x8)

    model = Model(init, out, name='VGG16')

    return model

if __name__ == "__main__":
    # from keras.utils.visualize_util import plot

    vgg16 = create_vgg16()
    # vgg16.summary()

    # plot(vgg16, to_file="VGG16.png", show_shapes=True)
