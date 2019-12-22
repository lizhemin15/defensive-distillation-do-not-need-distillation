# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 22:34:35 2019
test adv attack
@author: x1c
"""
import keras
from keras import backend,losses
from keras.models import load_model 
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle
adv_suc=0
import argparse
import os
from keras import backend as K
import tensorflow as tf
parser = argparse.ArgumentParser(description='train_mnist')
parser.add_argument('--gpu_id', default='0',
                    help='gpu_id')
parser.add_argument('--start',default=0,
                    help='start_temperature')
parser.add_argument('--end',default=100,
                    help='end_temperature')
parser.add_argument('--temper',default='1',
                    help='end_temperature')
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)

def fgsm(model,image,y_true,eps=0.1):
    y_pred=model.output
    loss=losses.categorical_crossentropy(y_true,y_pred)
    gradient=backend.gradients(loss,model.input)
    gradient=gradient[0]
    adv=image+backend.sign(gradient)*eps
    sess=backend.get_session()
    adv=sess.run(adv,feed_dict={model.input:image})
    adv=np.clip(adv,0,1)
    return adv

def cal_max_gradient(x_train,y_train,model_path):
    gradient=0
    keras.backend.set_learning_phase(0)
    kmodel = load_model(model_path)
    kmodel.pop()
    #kmodel.summary()
    y_train=keras.utils.to_categorical(y_train, 10)
    for i in range(100):
        image=x_train[i]/255
        image=image.reshape(1,28,28,1)
        y_true=y_train[i]
        y_pred=kmodel.output
        loss=losses.categorical_crossentropy(y_true,y_pred)
        gradient=backend.gradients(loss,kmodel.input)
        gradient=gradient[0]
        gradient=sess.run(gradient,feed_dict={kmodel.input:image})
        gradient+=np.abs(np.max(gradient))        
    return gradient/100


def cal_max_logits(x_train,model_path):
    kmodel = load_model(model_path)
    kmodel.pop()
    logits_max=0
    for i in range(100):
        image=x_train[i]/255
        image=image.reshape(1,28,28,1)
        logits=sess.run(kmodel.output,feed_dict={kmodel.input:image})
        logits_max+=np.max(logits)
    logits_max/=100
    return logits_max

def cal_test_acc(x_test,y_test,model_path):
    kmodel = load_model(model_path)
    acc=0
    for i in range(10000):
        image=x_test[i]/255
        image=image.reshape(1,28,28,1)
        pre=sess.run(kmodel.output,feed_dict={kmodel.input:image})
        y_true=y_test[i]
        if np.argmax(pre)==np.argmax(y_true):
            acc+=1
    acc/=10000
    return(acc)
def cal_adv_acc(x_train,y_train,model_path):
    adv_suc=0
    keras.backend.set_learning_phase(0)
    kmodel = load_model(model_path)
    kmodel.pop()
    kmodel.add(keras.layers.Softmax(axis=-1))
    for i in range(100):
        image=x_train[i]/255
        image=image.reshape(1,28,28,1)
        y_true=y_train[i]
        adv=fgsm(kmodel,image,y_true,eps=0.3)
        pre=sess.run(kmodel.output,feed_dict={kmodel.input:adv})
        if np.argmax(y_true) != np.argmax(pre):
            adv_suc+=1
    return adv_suc/100

def sec_logits(x_train,model_path):
    kmodel = load_model(model_path)
    kmodel.pop()
    sess=backend.get_session()
    logits_max=0
    for i in range(100):
        image=x_train[i]/255
        image=image.reshape(1,28,28,1)
        logits=sess.run(kmodel.output,feed_dict={kmodel.input:image})
        logits[0,np.argmax(logits)]=np.min(logits)
        logits_max+=np.max(logits)
    logits_max/=100
    return logits_max

def change_num(temper,num):
    if temper==True:
        tem_str=str(num)
        epoch_str=''
    else:
        tem_str='1'
        epoch_str=str(num)
    return tem_str,epoch_str

if __name__=='__main__':
    temper=args.temper
    if temper=='1':
        temper=True
    else:
        temper=False
    start_num=np.int(args.start)
    end_num=np.int(args.end)
    '''
    if temper==True:
        start_num = 1
        end_num = 100
    else:
        start_num = 0
        end_num = 50
    '''
    
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train=keras.utils.to_categorical(y_train, 10)
    y_test=keras.utils.to_categorical(y_test, 10)
    '''
    #different_temperature_max_logits_test
    logits_max={'teacher':[],'student':[]}
    for num in range(start_num,end_num):
        tem_str,epoch_str = change_num(temper,num)
        model_path=tem_str+'teacher'+epoch_str+'.h5'
        logits_max['teacher'].append(cal_max_logits(x_train,model_path))
        model_path=tem_str+'student'+epoch_str+'.h5'
        logits_max['student'].append(cal_max_logits(x_train,model_path))
        print('finish logitsmax:',num)
    with open(str(start_num)+'temp_logits_max.txt', 'wb') as file_pi:
        pickle.dump(logits_max, file_pi)
        
    #different_temperature_second_max_logits_test
    logits_sec={'teacher':[],'student':[]}
    for num in range(start_num,end_num):
        tem_str,epoch_str = change_num(temper,num)
        model_path=tem_str+'teacher'+epoch_str+'.h5'
        logits_sec['teacher'].append(sec_logits(x_train,model_path))    
        model_path=tem_str+'student'+epoch_str+'.h5'
        logits_sec['student'].append(sec_logits(x_train,model_path)) 
        print('finish logitssec:',num)
    with open(str(start_num)+'temp_logits_sec.txt', 'wb') as file_pi:
        pickle.dump(logits_sec, file_pi)
    
    '''
    '''
    #different_temperature_test_acc
    
    test_acc={'teacher':[],'student':[]}
    for num in range(start_num,end_num):
        tem_str,epoch_str = change_num(temper,num)
        model_path=tem_str+'teacher'+epoch_str+'.h5'
        test_acc['teacher'].append(cal_test_acc(x_test,y_test,model_path))
        model_path=tem_str+'student'+epoch_str+'.h5'
        test_acc['student'].append(cal_test_acc(x_test,y_test,model_path))
        print('finish testacc:',num)
    with open(str(start_num)+'temp_test_acc.txt', 'wb') as file_pi:
        pickle.dump(test_acc, file_pi)
    '''
    
    #different temperature adv_acc
    
    adv_acc={'teacher':[],'student':[]}
    for num in range(start_num,end_num):
        tem_str,epoch_str = change_num(temper,num)
        model_path=tem_str+'teacher'+epoch_str+'.h5'
        adv_acc['teacher'].append(cal_adv_acc(x_train,y_train,model_path))
        model_path=tem_str+'student'+epoch_str+'.h5'
        adv_acc['student'].append(cal_adv_acc(x_train,y_train,model_path))
        print('finish advacc:',num)
    with open(str(start_num)+'temp_adv_acc.txt', 'wb') as file_pi:
        pickle.dump(adv_acc, file_pi)
        
         
    #defferent_temperature test_max_gradient
    max_gradient={'teacher':[],'student':[]}
    for num in range(start_num,end_num):
        tem_str,epoch_str = change_num(temper,num)
        model_path=tem_str+'teacher'+epoch_str+'.h5'
        max_gradient['teacher'].append(cal_max_gradient(x_train,y_train,model_path))
        model_path=tem_str+'student'+epoch_str+'.h5'
        max_gradient['student'].append(cal_max_gradient(x_train,y_train,model_path))
        print('finish maxgradient:',num)
    with open(str(start_num)+'temp_max_gradient.txt', 'wb') as file_pi:
        pickle.dump(max_gradient, file_pi)
    
    
    
