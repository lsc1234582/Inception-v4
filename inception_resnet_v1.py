from keras.layers import merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
import numpy as np

"""
Implementation of Inception-Residual Network v1 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.

Some additional details:
[1] Each of the A, B and C blocks have a 'scale_residual' parameter.
    The scale residual parameter is according to the paper. It is however turned OFF by default.

    Simply setting 'scale=True' in the create_inception_resnet_v1() method will add scaling.
"""

def inception_resnet_stem(input):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2, 2))(input)
    c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = Convolution2D(64, 3, 3, activation='relu', )(c)
    c = MaxPooling2D((3, 3), strides=(2, 2))(c)
    c = Convolution2D(80, 1, 1, activation='relu', border_mode='same')(c)
    c = Convolution2D(192, 3, 3, activation='relu')(c)
    c = Convolution2D(256, 3, 3, activation='relu', subsample=(2,2), border_mode='same')(c)
    return c

def inception_resnet_A(input, scale_residual=False):
    # Input is relu activation
    init = input

    ir1 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir2)

    ir3 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir3)
    ir3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir3)

    ir_merge = merge([ir1, ir2, ir3], concat_axis=1, mode='concat')

    ir_conv = Convolution2D(256, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def inception_resnet_B(input, scale_residual=False):
    # Input is relu activation
    init = input

    ir1 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(128, 1, 7, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(128, 7, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=1)

    ir_conv = Convolution2D(896, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def inception_resnet_C(input, scale_residual=False):
    # Input is relu activation
    init = input

    ir1 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(192, 1, 3, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(192, 3, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=1)

    ir_conv = Convolution2D(1792, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def reduction_A(input, k=192, l=224, m=256, n=384):
    r1 = MaxPooling2D((3,3), strides=(2,2))(input)

    r2 = Convolution2D(n, 3, 3, activation='relu', subsample=(2,2))(input)

    r3 = Convolution2D(k, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(l, 3, 3, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(m, 3, 3, activation='relu', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=1)
    return m


def reduction_resnet_B(input):
    r1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)

    r2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r2 = Convolution2D(384, 3, 3, activation='relu', subsample=(2,2))(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(256, 3, 3, activation='relu', subsample=(2, 2))(r3)

    r4 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(r4)
    r4 = Convolution2D(256, 3, 3, activation='relu', subsample=(2, 2))(r4)

    m = merge([r1, r2, r3, r4], concat_axis=1, mode='concat')
    return m

def create_inception_resnet_v1(input, nb_output=1000, scale=False):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_resnet_stem(input)

    # 5 x Inception Resnet A
    x = inception_resnet_A(x, scale_residual=scale)
    x = inception_resnet_A(x, scale_residual=scale)
    x = inception_resnet_A(x, scale_residual=scale)
    x = inception_resnet_A(x, scale_residual=scale)
    x = inception_resnet_A(x, scale_residual=scale)

    # Reduction A - From Inception v4
    x = reduction_A(x, k=192, l=192, m=256, n=384)

    # 10 x Inception Resnet B
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)

    # Reduction Resnet B
    x = reduction_resnet_B(x)

    # 5 x Inception Resnet C
    x = inception_resnet_C(x, scale_residual=scale)
    x = inception_resnet_C(x, scale_residual=scale)
    x = inception_resnet_C(x, scale_residual=scale)
    x = inception_resnet_C(x, scale_residual=scale)
    x = inception_resnet_C(x, scale_residual=scale)

    # Average Pooling
    x = AveragePooling2D((7,7))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    x = Dense(output_dim=nb_output, activation='softmax')(x)
    return x

if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model
    from keras.utils.visualize_util import plot

    ip = Input(shape=(3, 299, 299))

    inception_resnet_v1 = create_inception_resnet_v1(ip, scale=False)
    model = Model(ip, inception_resnet_v1)

    plot(model, to_file="Inception ResNet-v1.png", show_shapes=True)