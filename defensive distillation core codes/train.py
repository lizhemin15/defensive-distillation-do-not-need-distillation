# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:52:39 2019

@author: x1c
"""
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D,Flatten,Dropout, Activation
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import numpy as np
from keras.optimizers import SGD,Adam
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import load_model
import tensorflow as tf
import os
import pickle
import argparse
import time

tf.reset_default_graph()

parser = argparse.ArgumentParser(description='train_mnist')
parser.add_argument('--gpu_id', default='0',
                    help='gpu_id')
parser.add_argument('--epochs',default=500,
                    help='epochs')
parser.add_argument('--lr',default=0.01,
                    help='lr')
parser.add_argument('--ls',default=0.01,
                    help='loss')
parser.add_argument('--data',default='cifar100',
                    help='dataset, include mnist, cifar10, cifar100')
parser.add_argument('--da',default=0,
                    help='0 means False, 1 means True')
parser.add_argument('--verbose',default=0,
                    help='0 means False, 1 means True')
parser.add_argument('--tem',default=1,
                    help='temperature')
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)


class models(object):
    def __init__(self,model_name,temperature=1,lr=0.01):
        self.model_name=model_name
        self.temperature=temperature
        self.lr = lr
        
    def load(self):
        if self.model_name == 'mnist':
            return self.load_mnist()
        elif self.model_name == 'cifar10':
            return self.load_cifar10()
        elif self.model_name == 'cifar100':
            return self.load_cifar100()
        else:
            raise('please inoput right dataset name, only include mnist, cifar10, cifar 100')
        
    def load_mnist(self):
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
        model.add(keras.layers.Softmax_t(axis=-1,temperature=self.temperature))
        sgd = optimizers.SGD(lr=self.lr, momentum=0.5)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model
    
    def load_cifar10(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3),  activation = 'relu',  input_shape=(32,32,3)))
        model.add(Conv2D(64, (3, 3),  activation = 'relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3),  activation = 'relu'))
        model.add(Conv2D(128, (3, 3),  activation = 'relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(10))
        #model.add(keras.layers.Softmax(axis=-1))
        model.add(keras.layers.Softmax_t(axis=-1,temperature=self.temperature))
        sgd = optimizers.SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model
    
    def load_cifar100(self):
        import resnet
        # input image dimensions
        img_rows, img_cols = 32, 32
        # The CIFAR100 images are RGB.
        img_channels = 3
        nb_classes = 100
        model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
        #model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)
        #model.add(keras.layers.Softmax_t(axis=-1,temperature=self.temperature))
        sgd = optimizers.SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #print(model.summary())
        return model
    
    '''
    def load_cifar100(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3),  activation = 'relu',  input_shape=(32,32,3)))
        model.add(Conv2D(64, (3, 3),  activation = 'relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3),  activation = 'relu'))
        model.add(Conv2D(128, (3, 3),  activation = 'relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(100))
        #model.add(keras.layers.Softmax(axis=-1))
        model.add(keras.layers.Softmax_t(axis=-1,temperature=self.temperature))
        sgd = optimizers.SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model
    '''


class train(object):
    def __init__(self,name,give=False,y_train=None,temperature=1,epochs=50,lr=0.01,verbose=False,ls=0.01):
        self.name = name
        self.give = give
        self.y_train = y_train
        self.temperature = temperature
        self.epochs = epochs
        self.lr = lr
        self.models = models(name,temperature,lr)
        self.verbose = verbose
        self.ls = ls

    
    def load_data(self):
        if self.name == 'cifar10':
            from keras.datasets import cifar10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_train=keras.utils.to_categorical(y_train, 10)
            y_test=keras.utils.to_categorical(y_test, 10)
        elif self.name == 'cifar100':
            from keras.datasets import cifar100
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            y_train=keras.utils.to_categorical(y_train, 100)
            y_test=keras.utils.to_categorical(y_test, 100)
        elif self.name == 'mnist':
            from keras.datasets import mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = np.expand_dims(x_train,axis=-1)
            x_test = np.expand_dims(x_test,axis=-1)
            y_train=keras.utils.to_categorical(y_train, 10)
            y_test=keras.utils.to_categorical(y_test, 10)
        else:
            raise('please cheak your dataset name, the valid dataset include cifar10, cifar100, mnist')
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        return x_train/255-0.5,y_train,x_test/255-0.5,y_test
    

    def train(self):
        # load data
        x_train, y_train0,x_test, y_test = self.load_data()

        if not self.give:
            self.y_train=y_train0
    
        # build network
        model = self.models.load()
        print(model.summary())

        #lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        #early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        #csv_logger = CSVLogger('resnet18_cifar10.csv')
        history_dict={'acc':[],'val_acc':[],'loss':[],'val_loss':[]}
        hist_names=['acc','val_acc','loss','val_loss']
        if int(args.da) == 1:
            for i in range(self.epochs):
                if i%10==0:
                    print('start train with da epoch:',i)
                train_datagen = keras.preprocessing.image.ImageDataGenerator(
                                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                height_shift_range=0.1,
                                horizontal_flip=True)
                test_datagen = keras.preprocessing.image.ImageDataGenerator()
                train_generator=train_datagen.flow(x_train, self.y_train, batch_size=128)
                test_generator=test_datagen.flow(x_test,y_test,batch_size=128)
                history=model.fit_generator(train_generator,
                                    steps_per_epoch=len(x_train) // 128, 
                                    epochs=1,
                                    validation_data=test_generator,
                                    validation_steps=len(x_test) // 128,
                                    shuffle=True,
                                    verbose=self.verbose,)
                                    #callbacks=[lr_reducer, early_stopper, csv_logger])
                for hist_name in hist_names:
                    history_dict[hist_name].append(history.history[hist_name])
                if history.history['loss'][0]<self.ls:
                    break
                
        else:
            for i in range(self.epochs):
                if i%10==0:
                    print('start train epoch:',i)
                history=model.fit(x_train, self.y_train,
                          batch_size=128,
                          epochs=1,
                          validation_data=(x_test, y_test),
                          shuffle=True,
                          verbose=self.verbose)
                        #callbacks=[lr_reducer, early_stopper, csv_logger],
                for hist_name in hist_names:
                    history_dict[hist_name].append(history.history[hist_name])
                if history.history['loss'][0]<self.ls:
                    break
        print('finish train temperature=',self.temperature)
        save_name = self.name+str(self.temperature)
        model.save(save_name+'.h5')
        with open(save_name+'.txt', 'wb') as file_pi:
            pickle.dump(history_dict, file_pi)
        return save_name

    def dis(self):
        raise('in the future do this')
        print('start train teacher with T:',self.temperature,'lr:',self.lr)
        save_name=self.train()
        x_train, y_train0,x_test, y_test = self.load_data()
        #x_train,y_train0,x_val,y_val=train_val(x_train,y_train0)
        model_teacher=load_model(save_name+'.h5')
        y_new=model_teacher.predict(x_train)
        print('start train student with T:',self.temperature,'lr:',self.lr)
        self.train()

if __name__=='__main__':
    stt = time.clock()
    temper_list=[1,2,3,4,5,6,7,8,9,10,11,12,20,40,60,80,100]
    temper_list=[float(args.tem)]
    for tem in temper_list:
        print('start train ',tem)
        my_train = train(name=str(args.data),
                         epochs=np.int(args.epochs),
                         lr=np.float(args.lr),
                         temperature=tem,
                         ls=np.float(args.ls),
                         verbose=int(args.verbose))
        my_train.train()
    print('time:',time.clock()-stt)
    
    
