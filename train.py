# -*- coding:utf-8 -*-
# Author:Richard Fang


from keras.datasets import cifar100
from keras.models import Model
from model import mobilenetv3

from keras import optimizers
from keras import metrics
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator


import os
from time import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
import cv2
import functools


def train(model, batch, epoch, X_train, y_train, X_test, y_test, data_augmentation=True):
    start = time()
    log_dir = datetime.now().strftime('model_%Y%m%d_%H%M')
    os.mkdir(log_dir)

    checkpoint = ModelCheckpoint(log_dir+ 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1)

    if data_augmentation:
        #  Using the data Augmentation in traning data
        datagen1 = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        datagen2 = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rescale=1. / 255)

        train_gen = datagen1.flow(x=X_train, y=y_train, batch_size=batch)
        test_gen = datagen2.flow(x=X_test, y=y_test, batch_size=batch)

        h = model.fit_generator(generator=train_gen,
                                steps_per_epoch=10000 / batch,
                                epochs=epoch,
                                validation_data=test_gen,
                                validation_steps=1000 / batch,
                                callbacks=[checkpoint, reduce_lr, early_stopping])




    print('\n@ Total Time Spent: %.2f seconds' % (time() - start))
    acc, val_acc = h.history['accuracy'], h.history['val_accuracy']
    m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    return h


def resize_cafir100(x):

    set = np.zeros((x.shape[0], 224, 224, 3))
    for i in tqdm(range(x.shape[0])):
        src = x[i]
        # print(src.dtype)
        dst = cv2.resize(src, dsize=(224, 224))
        set[i] = dst

    return set


if __name__ == '__main__':

    nb_classes = 10

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    X_train = X_train.astype('float32')[0:5000, :, :, :]
    X_test = X_test.astype('float32')[0:1000, :, :, :]

    X_train = resize_cafir100(X_train)
    X_test = resize_cafir100(X_test)

    y_train = to_categorical(y_train, nb_classes)[0:5000, :]
    y_test = to_categorical(y_test, nb_classes)[0:1000, :]


    epoch = 200
    batch = 128

    mnetv3 = mobilenetv3.Mobilenetv3_small(input_shape=(224, 224, 3), alpha=1.0, include_top=True, classes=10)

    # initiate RMSprop optimizer
    # 均方根反向传播（RMSprop，root mean square prop）优化
    # opt = optimizers.rmsprop(lr=0.001, decay=1e-6)
    opt = optimizers.SGD(lr=0.1, momentum=0.9, decay=0.01/3, nesterov=False)


    #
    top5_acc = functools.partial(metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = 'top5_acc'
    # Let's train the model using RMSprop
    # 使用均方根反向传播（RMSprop）训练模型
    mnetv3.compile(loss='categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy', top5_acc])

    h = train(mnetv3, batch, epoch,  X_train, y_train, X_test, y_test,)

