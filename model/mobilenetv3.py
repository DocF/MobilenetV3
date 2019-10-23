# -*- coding:utf-8 -*-
# Author:Richard Fang


from keras.models import Model
from keras.layers import *
import keras.backend as K


def relu6(x):

    return K.relu(x, max_value=6)


def hard_swish(x):

    return x * K.relu(x+3, max_value=6)/6


def return_activation(x, nl, name=None):

    if nl == 'HS':
        x = Activation(hard_swish, name=name+'HS')(x)
    else:
        x = Activation(relu6, name=name+'relu6')(x)

    return x


def squeeze_excite(inputs, name=None):

    # Paper words:
    # "Instead, we replace them all to fixed to be 1/4
    # of the number of channels in expansion layer."

    channels = K.int_shape(inputs)[-1]

    x = GlobalAveragePooling2D(name=name+'GAP')(inputs)
    x = Dense(channels//4, activation='relu', name=name+'FC1')(x)
    x = Dense(channels, activation='hard_sigmoid', name=name+'FC2')(x)
    x = Reshape((1, 1, channels))(x)
    x = Multiply(name=name+'Multiply')([inputs, x])

    return x


def conv_block(inputs, filters, kernel_size, strides, nl='HS', name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same' if strides == (1, 1) else 'valid',
               use_bias=False)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    x = return_activation(x, nl, name='Conv_Block_'+name+'_')

    return x


def conv_block_NBN(inputs, filters, kernel_size, strides, nl='HS', name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same' if strides == (1, 1) else 'valid',
               use_bias=False)(inputs)
    x = return_activation(x, nl, name='Conv_Block_NBN'+name+'_')

    return x


def blockneck(inputs, expand_size, out_channels, kernel_size, se, nl, strides, block_id, alpha=1.0):
    """
    Build the blockneck model of MobileNetV3.

    # Augments
        inputs: inputs tensor
        expand_size: the channels of expansion layer
        out_channels: the out channels of this blockneck
        kernel_size: Depthwise convolution kernel size
        se: boolean type, add a squeeze and excite model if se is 'True'
        nl: non-linearity
        strides: Depthwise convolution kernel strides
        block_id: the id of this blockneck
        alpha: width multiplier

    # Return
        A Keras layer
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    in_channels = K.int_shape(inputs)[channel_axis]
    out_channels = int(out_channels * alpha)
    x = inputs
    prefix = 'Block{}_'.format(block_id)

    # Expand
    x = Conv2D(filters=expand_size,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same' if strides == (1, 1) else 'valid',
               use_bias=False,
               name=prefix + 'Expand')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'Expand_BN')(x)
    x = return_activation(x, nl, prefix + 'Expand_')

    # Depthwise
    x = DepthwiseConv2D(kernel_size=kernel_size,
                        strides=strides,
                        padding='same' if strides == (1, 1) else 'valid',
                        use_bias=False,
                        name=prefix + 'Depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'Depthwise_BN')(x)
    x = return_activation(x, nl, prefix + 'Depthwise_')

    # Squeeze and Excite
    if se:
        x = squeeze_excite(x, name=prefix+'SE_')

    # Project
    x = Conv2D(filters=out_channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same' if strides == (1, 1) else 'valid',
               use_bias=False,
               name=prefix+'Project')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'Project_BN')(x)

    # Shortcut
    if in_channels == out_channels and strides == (1, 1):
        x = Add(name=prefix + 'Add')([inputs, x])

    return x


def Mobilenetv3_small(input_shape=None,
                      alpha=1.0,
                      include_top=True,
                      classes=1000,
                      **kwargs):
    """
    Instantiates the MobileNetV3-small architecture.

    # Arguments
        input_shape: optional shape tuple， shape of input tensor.
        alpha: width multiplier.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        classes: the number of class, only to be specified if `include_top` is True


    # Return
        A Keras model instance.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    inputs = Input(shape=input_shape)

    x = conv_block(inputs, filters=16, kernel_size=(3, 3), strides=(2, 2), nl='HS', name='1')

    x = blockneck(inputs=x, expand_size=16, out_channels=16, kernel_size=(3, 3),
                  se=True, nl='RE', strides=(2, 2), alpha=alpha, block_id=1)

    x = blockneck(inputs=x, expand_size=72, out_channels=24, kernel_size=(3, 3),
                  se=False, nl='RE', strides=(2, 2), alpha=alpha, block_id=2)

    x = blockneck(inputs=x, expand_size=88, out_channels=24, kernel_size=(3, 3),
                  se=False, nl='RE', strides=(1, 1), alpha=alpha, block_id=3)
    x = blockneck(inputs=x, expand_size=96, out_channels=40, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(2, 2), alpha=alpha, block_id=4)

    x = blockneck(inputs=x, expand_size=240, out_channels=40, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=5)
    x = blockneck(inputs=x, expand_size=240, out_channels=40, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=6)
    x = blockneck(inputs=x, expand_size=120, out_channels=48, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=7)
    x = blockneck(inputs=x, expand_size=144, out_channels=48, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=8)

    x = blockneck(inputs=x, expand_size=288, out_channels=96, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(2, 2), alpha=alpha, block_id=9)
    x = blockneck(inputs=x, expand_size=576, out_channels=96, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=10)
    x = blockneck(inputs=x, expand_size=576, out_channels=96, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=11)
    x = conv_block(inputs=x, filters=576, kernel_size=(1, 1), strides=(1, 1), nl='HS', name='2')

    x = GlobalAveragePooling2D()(x)
    channels = K.int_shape(x)[channel_axis]
    x = Reshape((1, 1, channels))(x)
    x = conv_block_NBN(inputs=x, filters=1280, kernel_size=(1, 1), strides=(1, 1), nl='HS', name='1')

    if include_top:
        x = Conv2D(classes, (1, 1), padding='same', activation='softmax')(x)
        x = Reshape((classes,))(x)

    model = Model(inputs, x)

    return model


def Mobilenetv3_large(input_shape=None,
                      alpha=1.0,
                      include_top=True,
                      classes=1000,
                      **kwargs):
    """
    Instantiates the MobileNetV3-large architecture.

    # Arguments
        input_shape: optional shape tuple， shape of input tensor.
        alpha: width multiplier.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        classes: the number of class, only to be specified if `include_top` is True


    # Return
        A Keras model instance.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    inputs = Input(shape=input_shape)

    x = conv_block(inputs, filters=16, kernel_size=(3, 3), strides=(2, 2), nl='HS', name='1')

    x = blockneck(inputs=x, expand_size=16, out_channels=16, kernel_size=(3, 3),
                  se=False, nl='RE', strides=(1, 1), alpha=alpha, block_id=1)
    x = blockneck(inputs=x, expand_size=64, out_channels=24, kernel_size=(3, 3),
                  se=False, nl='RE', strides=(2, 2), alpha=alpha, block_id=2)

    x = blockneck(inputs=x, expand_size=72, out_channels=24, kernel_size=(3, 3),
                  se=False, nl='RE', strides=(1, 1), alpha=alpha, block_id=3)
    x = blockneck(inputs=x, expand_size=72, out_channels=40, kernel_size=(5, 5),
                  se=True, nl='RE', strides=(2, 2), alpha=alpha, block_id=4)

    x = blockneck(inputs=x, expand_size=120, out_channels=40, kernel_size=(5, 5),
                  se=True, nl='RE', strides=(1, 1), alpha=alpha, block_id=5)
    x = blockneck(inputs=x, expand_size=120, out_channels=40, kernel_size=(5, 5),
                  se=True, nl='RE', strides=(1, 1), alpha=alpha, block_id=6)
    x = blockneck(inputs=x, expand_size=240, out_channels=80, kernel_size=(3, 3),
                  se=True, nl='HS', strides=(2, 2), alpha=alpha, block_id=7)

    x = blockneck(inputs=x, expand_size=200, out_channels=80, kernel_size=(3, 3),
                  se=False, nl='HS', strides=(1, 1), alpha=alpha, block_id=8)
    x = blockneck(inputs=x, expand_size=184, out_channels=80, kernel_size=(3, 3),
                  se=False, nl='HS', strides=(1, 1), alpha=alpha, block_id=9)
    x = blockneck(inputs=x, expand_size=184, out_channels=80, kernel_size=(3, 3),
                  se=False, nl='HS', strides=(1, 1), alpha=alpha, block_id=10)
    x = blockneck(inputs=x, expand_size=480, out_channels=112, kernel_size=(3, 3),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=11)
    x = blockneck(inputs=x, expand_size=672, out_channels=112, kernel_size=(3, 3),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=12)
    x = blockneck(inputs=x, expand_size=672, out_channels=160, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=13)
    x = blockneck(inputs=x, expand_size=672, out_channels=160, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(2, 2), alpha=alpha, block_id=14)

    x = blockneck(inputs=x, expand_size=960, out_channels=160, kernel_size=(5, 5),
                  se=True, nl='HS', strides=(1, 1), alpha=alpha, block_id=15)
    x = conv_block(inputs=x, filters=960, kernel_size=(1, 1), strides=(1, 1), nl='HS', name='2')

    x = GlobalAveragePooling2D()(x)
    channels = K.int_shape(x)[channel_axis]
    x = Reshape((1, 1, channels))(x)
    x = return_activation(x, 'HS', 'GAP')
    x = conv_block_NBN(inputs=x, filters=1280, kernel_size=(1, 1), strides=(1, 1), nl='HS', name='1')

    if include_top:
        x = Conv2D(classes, (1, 1), padding='same', activation='softmax')(x)
        x = Reshape((classes,))(x)

    model = Model(inputs, x)

    return model


# if __name__ == '__main__':
#     input_shape = (416, 416, 3)
#     mnetv3 = Mobilenetv3_large(input_shape=input_shape, include_top=True)
#     # mnetv3.save('temp.h5')
#     mnetv3.summary()

