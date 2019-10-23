# -*- coding:utf-8 -*-
# Author:Richard Fang


from keras.datasets import cifar100
from keras.models import Model
from model import mobilenetv3



def train(dataset):
    if dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    mnetv3 = mobilenetv3.Mobilenetv3_large(input_shape=None,
                                           alpha=1.0,
                                           include_top=True,
                                           classes=100)
    mnetv3.compile(optimizer='rmsprop', loss='categorical_crossentropy')