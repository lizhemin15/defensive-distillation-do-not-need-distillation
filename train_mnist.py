# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:52:39 2019

@author: x1c
"""
import keras
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D,Flatten,Dropout
from keras.callbacks import LearningRateScheduler, TensorBoard
import numpy as np
from keras.optimizers import SGD,Adam
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import load_model
import tensorflow as tf
import os
import pickle
import argparse
epochs=50
parser = argparse.ArgumentParser(description='train_mnist')
parser.add_argument('--gpu_id', default='0',
                    help='gpu_id')
parser.add_argument('--start',default=0,
                    help='start_temperature')
parser.add_argument('--end',default=100,
                    help='end_temperature')
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)

def build_model(temperature,lr):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(28,28,1)))
    model.add(Conv2D(32, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(10, kernel_initializer='he_normal'))
    #model.add(keras.layers.Softmax(axis=-1))
    model.add(keras.layers.Softmax_t(axis=-1,temperature=temperature))
    sgd = optimizers.SGD(lr=lr, momentum=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def train(name,give=False,y_train=None,temperature=1,epochs=50,lr=0.1):
    # load data
    (x_train, y_train0), (x_test, y_test) = mnist.load_data()
    y_train0 = keras.utils.to_categorical(y_train0, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train0=y_train0.reshape(60000,10)
    y_test=y_test.reshape(10000,10)
    x_train=np.expand_dims(x_train,axis=-1)
    x_test=np.expand_dims(x_test,axis=-1)
    x_train /= 255.0
    x_test /= 255.0
    if not give:
        y_train=y_train0

    # build network
    model = build_model(temperature,lr)
    #print(model.summary())


    history=model.fit(x_train, y_train,
              batch_size=128,
              epochs=epochs,
              callbacks=[],
              validation_data=(x_test, y_test),
              shuffle=True,
              verbose=False)
    model.save(name+'.h5')
    with open(name+'.txt', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def dis(temperature=1,epochs=50,temper=True,lr=0.1):
    if temper==True:
        tem_str=str(temperature)
        epoch_str=''
    else:
        tem_str=''
        epoch_str=str(epochs)
    print('start train teacher with T:',temperature,'lr:',lr)
    train(tem_str+'teacher'+epoch_str+'lr_'+str(lr),temperature=temperature,epochs=epochs,lr=lr)
    (x_train, y_train0), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_train=np.expand_dims(x_train,axis=-1)
    x_train /= 255.0
    model_teacher=load_model(tem_str+'teacher'+epoch_str+'lr_'+str(lr)+'.h5')
    y_new=model_teacher.predict(x_train)
    print('start train student with T:',temperature,'lr:',lr)
    train(tem_str+'student'+epoch_str+'lr_'+str(lr),give=True,y_train=y_new,temperature=temperature,epochs=epochs,lr=lr)

if __name__=='__main__':
    '''
    for i in range(np.int(args.start),np.int(args.end)):
        print('start training temoerature of ',str(i))
        dis(temperature=1,epochs=i,temper=True)
        print('finish training temoerature of ',str(i))
    '''
    temper_list=[1,2,3,4,5,6,7,8,9,10,11,12,20,40,60,80,100]
    for tem in temper_list:
        dis(temperature=tem,lr=0.1*tem)

    
