# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:09:35 2020

@author: x1c
"""

import matplotlib.pyplot as plt
import numpy as np
import train

data_name = ['mnist','cifar10']
nrows=4
ncols=5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='all', )
ax = ax.flatten()
ax_i=0
for name in data_name:
    my_train=train.train(name)
    x_train,y_train,x_test,y_test = my_train.load_data()
    x_train=(x_train+0.5)*255
    x_test=(x_test+0.5)*255
    x_train=x_train.astype(np.uint8)
    x_test=x_test.astype(np.uint8)
    class_num = y_test.shape[1]
    #class_num = 10
    plot_i = 0
    for ite_class in range(class_num):
        while np.argmax(y_train[plot_i,:])!= ite_class:
            plot_i+=1
        if name == 'mnist':
            ax[ax_i].imshow(x_train[plot_i,:,:,0], cmap='Greys', interpolation='nearest')
        else:
            ax[ax_i].imshow(x_train[plot_i,:,:,:],  interpolation='nearest')
        ax_i+=1
        if ax_i>=nrows*ncols:
            break

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig('./samples.png',pad_inches=0.0)
plt.show()
