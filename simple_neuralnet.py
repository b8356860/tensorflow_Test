# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 00:21:56 2016

@author: siang
"""

import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    
    wx_plus_b = tf.matmul(inputs,weights)+biases
    if activation_function == None:
        output = wx_plus_b
    else:
        output = activation_function(wx_plus_b)
    return output
    
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise =np.random.normal(0,0.05, x_data.shape)
y_data = np.square(x_data)- 0.5 + noise

xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])



    