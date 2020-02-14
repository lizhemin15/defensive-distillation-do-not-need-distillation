# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 22:34:35 2019
test adv attack
@author: x1c
"""
import keras
from keras import backend,losses
from keras.models import load_model 
import numpy as np
import matplotlib.pyplot as plt
import pickle
adv_suc=0
import argparse
import os
from keras import backend as K
import tensorflow as tf
from keras.models import Model

parser = argparse.ArgumentParser(description='adv_box')
parser.add_argument('--gpu_id', default='0',
                    help='gpu_id')
parser.add_argument('--mode',default='adv',
                    help='mode,include adv, grad ,logits')
parser.add_argument('--big',default=False,
                    help='bool value,if Ture, test from big to small')
parser.add_argument('--data',default='cifar10',
                    help='dataset, include mnist, cifar10, cifar100')
args = parser.parse_args()
print(args)




def initial():
    global sess
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)

def image_reshape(image):
    name = str(args.data)
    if name == 'mnist':
        image = image.reshape(1,28,28,1)
    else:
        image = image.reshape(1,32,32,3)
    return image

def fgsm(model,image,y_true,eps=0.03):
    #tf.reset_default_graph()
    y_pred=model.output
    loss=losses.categorical_crossentropy(y_true,y_pred)
    gradient=backend.gradients(loss,model.input)
    gradient=gradient[0]
    adv=image+backend.sign(gradient)*eps
    adv=sess.run(adv,feed_dict={model.input:image})
    adv=np.clip(adv,-0.5,0.5)
    #print('adv now:',np.max(adv-image))
    #tf.get_default_graph().finalize()
    return adv

def cal_max_gradient(x_train,y_train,model_path):
    gradient_out=0
    keras.backend.set_learning_phase(0)
    kmodel = load_model(model_path)
    if str(args.data)=='cifar100':
        pre = keras.layers.Softmax(axis=-1)(kmodel.get_layer(index=-2).output)
        kmodel = Model(inputs=kmodel.input,output=pre)
    else:
        kmodel.pop()
        kmodel.add(keras.layers.Softmax(axis=-1))
    kmodel.summary()
    #y_train=keras.utils.to_categorical(y_train, 10)
    for i in range(100):
        image=x_train[i]/255
        image-=0.5
        image=image_reshape(image)
        y_true=y_train[i]
        y_pred=kmodel.output
        loss=losses.categorical_crossentropy(y_true,y_pred)
        gradient=backend.gradients(loss,kmodel.input)
        gradient=gradient[0]
        gradient=sess.run(gradient,feed_dict={kmodel.input:image})
        gradient_out+=np.abs(np.max(gradient))
        #tf.reset_default_graph()
        #tf.get_default_graph().finalize()
        print(gradient_out)


    return gradient_out/100


def cal_max_logits(x_train,model_path):
    kmodel = load_model(model_path)
    if str(args.data)=='cifar100':
        kmodel = Model(inputs=kmodel.input,output=kmodel.get_layer(index=-2).output)
    else:
        kmodel.pop()
    logits_max=0
    for i in range(100):
        #tf.reset_default_graph()
        image=x_train[i]/255
        image-=0.5
        image=image_reshape(image)
        logits=sess.run(kmodel.output,feed_dict={kmodel.input:image})
        logits_max+=np.max(logits)
        #tf.get_default_graph().finalize()
    logits_max/=100
    return logits_max

def cal_test_acc(x_test,y_test,model_path):
    kmodel = load_model(model_path)
    acc=0
    for i in range(10000):
        #tf.reset_default_graph()
        image=x_test[i]/255
        image-=0.5
        image=image_reshape(image)
        pre=sess.run(kmodel.output,feed_dict={kmodel.input:image})
        y_true=y_test[i]
        if np.argmax(pre)==np.argmax(y_true):
            acc+=1
        #tf.get_default_graph().finalize()
    acc/=10000
    return(acc)
def cal_adv_acc(x_train,y_train,model_path):
    adv_suc=0
    keras.backend.set_learning_phase(0)
    kmodel = load_model(model_path)
    #print(kmodel.summary())
    if str(args.data)=='cifar100':
        pre = keras.layers.Softmax(axis=-1)(kmodel.get_layer(index=-2).output)
        kmodel = Model(inputs=kmodel.input,output=pre)
    else:
        kmodel.pop()
        kmodel.add(keras.layers.Softmax(axis=-1))
    #kmodel.summary()
    for i in range(100):
        #tf.reset_default_graph()
        image=x_train[i]/255
        image-=0.5
        image=image_reshape(image)
        y_true=y_train[i]
        adv=fgsm(kmodel,image,y_true,eps=0.03)
        adv_pre=sess.run(kmodel.output,feed_dict={kmodel.input:adv})
        ori_pre=sess.run(kmodel.output,feed_dict={kmodel.input:image})
        ori_pre=np.argmax(ori_pre)
        adv_pre=np.argmax(adv_pre)
        y_true=np.argmax(y_true)
        #print('true:%d , prediction:%d , adv:%d' % (y_true,ori_pre,adv_pre))
        if  ori_pre != adv_pre:
            adv_suc+=1
            print('ok')
        #tf.get_default_graph().finalize()
    print(adv_suc)
    return adv_suc/100

def cal_sec_logits(x_train,model_path):
    kmodel = load_model(model_path)
    if str(args.data)=='cifar100':
        kmodel = Model(inputs=kmodel.input,output=kmodel.get_layer(index=-2).output)
    else:
        kmodel.pop()
        #kmodel.add(keras.layers.Softmax(axis=-1))
    sess=backend.get_session()
    logits_max=0
    for i in range(100):
        #tf.reset_default_graph()
        image=x_train[i]/255
        image-=0.5
        image=image_reshape(image)
        logits=sess.run(kmodel.output,feed_dict={kmodel.input:image})
        logits[0,np.argmax(logits)]=np.min(logits)
        logits_max+=np.max(logits)
        #tf.get_default_graph().finalize()
    logits_max/=100
    return logits_max


class test(object):
    def __init__(self,temper_list,x_train,y_train,x_test,y_test,name):
        self.temper_list = temper_list
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.data_name = name
    
    def cal_logits(self):
        #different_temperature_second_max_logits_test
        logits={'max':[],'sec':[]}
        #for num in range(start_num,end_num):
        for num in self.temper_list:
            initial()
            model_path=self.data_name+str(num)+'.h5'
            logits['max'].append(cal_max_logits(self.x_train,model_path)) 
            logits['sec'].append(cal_sec_logits(self.x_train,model_path))    
            print('finish logits:',num,', is:',logits['max'][-1]-logits['sec'][-1])
            tf.reset_default_graph()
        with open(self.data_name+'_logits.txt', 'wb') as file_pi:
            pickle.dump(logits, file_pi)
    
    def cal_grad(self):
        #defferent_temperature test_max_gradient
        max_gradient=[]
        #for num in range(start_num,end_num):
        for num in self.temper_list:
            initial()
            print('start magradient:',num)
            model_path=self.data_name+str(num)+'.h5'
            max_gradient.append(cal_max_gradient(self.x_train,self.y_train,model_path))
            print('finish maxgradient:',num)
            tf.reset_default_graph()
        with open(self.data_name+'_max_gradient.txt', 'wb') as file_pi:
            pickle.dump(max_gradient, file_pi)
    
    def cal_adv(self):
        #different temperature adv_acc
        adv_acc=[]
        #for num in range(start_num,end_num):
        for num in self.temper_list:
            initial()
            model_path=self.data_name+str(num)+'.h5'
            adv_acc.append(cal_adv_acc(self.x_train,self.y_train,model_path))
            print('finish advacc:',num)
            tf.reset_default_graph()
        with open(self.data_name+'_adv_acc.txt', 'wb') as file_pi:
            pickle.dump(adv_acc, file_pi)

def load_data(name):
    if name == 'cifar10':
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train=keras.utils.to_categorical(y_train, 10)
        y_test=keras.utils.to_categorical(y_test, 10)
    elif name == 'cifar100':
        from keras.datasets import cifar100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        y_train=keras.utils.to_categorical(y_train, 100)
        y_test=keras.utils.to_categorical(y_test, 100)
    elif name == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train=keras.utils.to_categorical(y_train, 10)
        y_test=keras.utils.to_categorical(y_test, 10)
    else:
        raise('please cheak your dataset name, the valid dataset include cifar10, cifar100, mnist')
    return x_train,y_train,x_test,y_test

if __name__=='__main__':
    #temper_list=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[20,40,60,80,100]]
    temper_list=[1,2,3,4,5,6,7,8,9,10,11,12,20,40,60,80,100]
    temper_list=[17.678]
    if bool(args.big) == True:
        temper_list=temper_list[::-1]
    #temper_list=[20,40,60,80,100]
    data_name = str(args.data)
    x_train,y_train,x_test,y_test=load_data(name=data_name)
    test_box = test(temper_list,x_train,y_train,x_test,y_test,name=data_name)
    mode = str(args.mode)
    if mode=='adv':
        test_box.cal_adv()
    elif mode == 'grad':
        test_box.cal_grad()
    elif mode == 'logits':
        test_box.cal_logits()
    else:
        print('invalid value of mode, you only can input "grad","logits","adv"')

    

    
    
    
