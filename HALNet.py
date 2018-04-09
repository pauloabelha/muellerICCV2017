from __future__ import division

import io_model

from keras.utils import plot_model

from keras.models import Model
from keras.layers import (
    Input,
    Activation
)
from keras.layers.convolutional import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D
)
from keras import optimizers
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import objectives


def _deconv_layer(kernel_size, filters, stride, name):
    def f(input):
        elem = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                        strides=stride, padding='same', name=name)(input)
        return elem
    return f

def _interm_heatmap_loss(kernel_size, stride, num_heatmaps, parent_name='root', name_suffix='1', block_name='interm_hm_loss'):

    def f(input):
        elem1 = _deconv_layer(kernel_size=kernel_size, stride=stride,
                      filters=num_heatmaps,
                      name=parent_name + '_' + block_name +
                           '_' + name_suffix + '_'+'deconv')(input)
        return elem1
    return f

def _conv_block(kernel_size, stride,
                num_feature_maps, parent_name='root',
                name_suffix='1', block_name='conv'):
    """Builds a convolution layer.
    """

    def f(input):
        elem1 = Conv2D(filters=num_feature_maps, kernel_size=kernel_size,
               strides=stride, padding="same",
                       name=parent_name+'_'
                            +block_name+'_'+
                            name_suffix+'_'+'convLayer')(input)
        elem2 = BatchNormalization(scale=True, name=parent_name+'_'+block_name+'_'+name_suffix+'_'+'bnScale')(elem1)

        return elem2

    return f

def _res_block_ident_skip(f1, f2, parent_name='root', block_name='resIdskip'):
    """Builds a residual block (identity skip).
    """
    def f(input):
        block1 = _conv_block(1, 1, f1, parent_name=parent_name+'_'+block_name, name_suffix='1')(input)
        block1 = Activation('relu', name=parent_name+'_'+block_name+'_relu1')(block1)
        block2 = _conv_block(3, 1, f1, parent_name=parent_name+'_'+block_name, name_suffix='2')(block1)
        block2 = Activation('relu', name=parent_name+'_'+block_name+'_relu2')(block2)
        block3 = _conv_block(1, 1, f2, parent_name=parent_name+'_'+block_name, name_suffix='3')(block2)
        elmt_wise = add([input, block3])
        return Activation('relu', name=parent_name+'_'+block_name+'_relu3')(elmt_wise)

    return f

def _res_block_conv(stride, f1, f2, parent_name='root', block_name='resConv'):
    """Builds a residual block (convolution).
    """
    def f(input):
        block1 = _conv_block(1, stride, f1, parent_name=parent_name+'_'+block_name, name_suffix='1')(input)
        block1 = Activation('relu', name=parent_name+'_'+block_name+'_relu1')(block1)
        block2 = _conv_block(3, 1, f1, parent_name=parent_name+'_'+block_name, name_suffix='2')(block1)
        block2 = Activation('relu', name=parent_name+'_'+block_name+'_relu2')(block2)
        block3 = _conv_block(1, 1, f2, parent_name=parent_name+'_'+block_name, name_suffix='3')(block2)
        shortcut = _conv_block(1, stride, f2, parent_name=parent_name+'_'+block_name, name_suffix='4')(input)
        elmt_wise = add([shortcut, block3], name=parent_name+'_'+block_name+'_add')
        return Activation('relu', name=parent_name+'_'+block_name+'_relu3')(elmt_wise)

    return f


# simple Euclidean distance to use as loss for the model
# https://www.codeday.top/2017/11/02/54486.html
def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def _build_HALNet():
    num_heatmaps = 1
    level_name = 'level'
    # build level 1
    level_ix = '1'
    input = Input(shape=(320, 240, 4,), name=level_name + level_ix + '_input')
    conv1 = _conv_block(7, 1, 64,
                        parent_name=level_name + level_ix,
                        block_name='conv')(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name='level1_mp')(conv1)
    # build level 2
    level_ix = '2'
    res2a = _res_block_conv(1, 64, 256, parent_name=level_name + level_ix, block_name='res2a')(pool1)
    res2b = _res_block_ident_skip(64, 256, parent_name=level_name + level_ix, block_name='res2b')(res2a)
    res2c = _res_block_ident_skip(64, 256, parent_name=level_name + level_ix, block_name='res2c')(res2b)
    # build level 3
    level_ix = '3'
    res3a = _res_block_conv(2, 128, 512, parent_name=level_name + level_ix, block_name='res3a')(res2c)
    res3b = _res_block_ident_skip(128, 512, parent_name=level_name + level_ix, block_name='res3b')(res3a)
    res3c = _res_block_ident_skip(128, 512, parent_name=level_name + level_ix, block_name='res3c')(res3b)
    # build level 4
    level_ix = '4'
    res4a = _res_block_conv(2, 256, 1024, parent_name=level_name + level_ix, block_name='res4a')(res3c)
    res4b = _res_block_ident_skip(256, 1024, parent_name=level_name + level_ix, block_name='res4b')(res4a)
    res4c = _res_block_ident_skip(256, 1024, parent_name=level_name + level_ix, block_name='res4c')(res4b)
    res4d = _res_block_ident_skip(256, 1024, parent_name=level_name + level_ix, block_name='res4d')(res4c)
    # build level 5
    level_ix = '5'
    conv4e = _conv_block(3, 1, 512, parent_name=level_name + level_ix, block_name='conv4e')(res4d)
    conv4f = _conv_block(3, 1, 256, parent_name=level_name + level_ix, block_name='conv4f')(conv4e)
    # build heatmap conversion block (part of main loss block)
    main_loss_conv = Conv2D(filters=num_heatmaps, kernel_size=3,
                         strides=1, padding="same",
                         name='main_loss_conv')(conv4f)
    main_output = Conv2DTranspose(filters=num_heatmaps, kernel_size=3,
                                      strides=1, padding='same', name='main_loss_deconv')(main_loss_conv)
    # get intermediate losses
    interm_output1 = Conv2DTranspose(filters=num_heatmaps, kernel_size=4,
                                     strides=4, padding='same',
                                     name='interm_output1_deconv')(res3a)
    interm_output2 = Conv2DTranspose(filters=num_heatmaps, kernel_size=4,
                                     strides=4, padding='same',
                                     name='interm_output2_deconv')(res4a)
    interm_output3 = Conv2DTranspose(filters=num_heatmaps, kernel_size=4,
                                     strides=4, padding='same',
                                     name='interm_output3_deconv')(conv4e)
    # return model
    model = Model(inputs=input,
                  outputs=[interm_output1, interm_output2,
                           interm_output3, main_output])
    with open('HALNet_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    plot_model(model, to_file='HALNet.png', show_shapes=True)
    io_model.save(model, 'HALNet')
    return model

def load(load_model=True):
    model = None
    if load_model:
        try:
            a = 1/0
            print("Loading HALNet from file...")
            model = io_model.load('HALNet')
        except:
            print("Could not load HALNet. Building a new one for you...")
            model = _build_HALNet()
    adaDelta = optimizers.Adadelta(lr=0.05, rho=0.9)
    model.compile(loss=[euclidean_distance_loss, euclidean_distance_loss,
                        euclidean_distance_loss, euclidean_distance_loss],
                  optimizer=adaDelta, metrics=['accuracy'])
    return model