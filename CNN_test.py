# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:40:35 2016

@author: siang
"""
import tensorflow as tf
import numpy as np
import MyecgError
import time
import platform
#import matplotlib.pyplot as plt
import sklearn.metrics as skm
import os

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv1d(x, W):
    return tf.nn.conv1d(x, W, 1, 'SAME')


def max_pool_2(x):
    x = x[:,np.newaxis,:,:]
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')[:,0,:,:]


def random_batch(example, label, batch_num=None):
    example_num = len(example)
    if batch_num is None:
        batch_num = int(example_num*(2/3))
    else:
        if batch_num > example_num:
           raise  MyecgError.MyecgInvalidDataError(message='batch number is great than example number')
    rand = np.random.random_sample(example_num)
    order = np.argsort(rand)
    return example[order[:batch_num]], label[order[:batch_num]]


def random_mix(amount, *args):
    """
    Mix data of several types by randomlize method
    parameter
    ---------
    amount : int
        the number of per type user want to mix in output data
    returns
    -------
    mixed_data : list
        target data after mixed
    remain_data : list
        data isn't in mixed_data
    """
    mixed_data = []
    remain_data = []
    if amount > 50:
        for count,data in enumerate(args):
            data = np.array(data)
            temp_data = []
            rand = np.random.random_sample(len(data))
            sort_order = np.argsort(rand)
            if (len(data)/2)==amount:
                temp_data.extend(list(data[sort_order[:int(len(data)/2)]]))
                remain_data.extend(list(data[sort_order[int(len(data)/2):]]))
            elif (len(data)/2)>amount:
                temp_data.extend(list(data[sort_order[:amount]]))
                remain_data.extend(list(data[sort_order[amount:]]))
            else:
                temp_data.extend(list(data[sort_order[:int(len(data)/2)]]))
                remain_data.extend(list(data[sort_order[int(len(data)/2):]]))         
                while(True):
                    vacancy = amount-len(temp_data)
                    if vacancy>=len(temp_data):
                        temp_data.extend(temp_data)
                    elif vacancy<=0:
                        break
                    else:
                        temp_data.extend(temp_data[:vacancy])
            mixed_data.extend(temp_data)
    elif amount <=1:
        for count,data in enumerate(args):
            data = np.array(data)
            rand = np.random.random_sample(len(data))
            sort_order = np.argsort(rand)
            
            mixed_data.extend(list(data[sort_order[:int(len(data)*amount)]]))
            remain_data.extend(list(data[sort_order[int(len(data)*amount):]]))
            
    return np.array(mixed_data), np.array(remain_data)


def normalize(array):
    arrmin = np.min(array)
    array = array-arrmin
    arrmax = np.max(array)
    return array/arrmax


def confusion_matrix(v_xs, v_ys, kp):
    global y_conv
    y_pre = sess.run(y_conv, feed_dict={x: x_testdata, keep_prob: kp})
    result = tf.argmax(y_pre, 1)
    true = tf.argmax(v_ys, 1)
    y_result , y_true= sess.run([result,true], feed_dict={x:v_xs, y_:v_ys})
    failue_index = np.where((y_result!=y_true))
    conf_matrix = skm.confusion_matrix(y_true, y_result)
    return conf_matrix, failue_index


def submit_report(model, train_data, test_data, time_cost, epoch, step, 
                  accuracy, train_errors, test_errors, 
                  confusion_matrix, failure_areas):
    """
    report the result of this training, text report format example as follows
    
    REPORT NO.number (system, machine name, release, version, machine, processor)
    TRAINING MODEL    : model
    TRAINING STEP     : step
    TRAINING EPOCH    : epoch
    TRAINING TIME     : time_cost
    TEST ACCURACY     : accuracy
    TEST FAILURE NUMBER: len(fialure_areas)
    CONFUSION MATRIX  :
        1    2    3    4    5    6
    1   150  0    0    0    0    0
    2   0    656  0    0    0    0
    3   0    0    37   0    0    0
    4   0    0    0   56    0    0
    5   0    0    0    0 1004   30
    6   0    0    0    0   24   34


    figure report
    1. traning errors
    2. test errors
    3. fialure_areas in fialure_areas folder
    """
    number = time.strftime('%y%m%d%H%M')
    report_path = os.path.join('..','reports', model, 'report'+number,'report'+number)
    os.makedirs(os.path.dirname(report_path))
    system_inform = platform.uname()
    failure_number = len(failure_areas)
    with open(report_path+'.txt', 'w') as report:
        report.write('REPORT NO.{0} {1}\n'.format(number, system_inform))
        report.write('TRAINING MODEL    : {0}\n'.format(model))
        report.write('TRAINING STEP     : {0}\n'.format(step))        
        report.write('TRAINING EPOCH    : {0}\n'.format(epoch))
        report.write('TRAINING TIME     : {0:.3f} sec\n'.format(time_cost))
        report.write('TEST ACCURACY     : {0:.3f}%\n'.format(accuracy*100))
        report.write('TEST FAILURE NUMBER: {0}\n'.format(failure_number))
        
        report.write('CONFUSION MATRIX  : \n')
        report.write('\t')
        for i in range(len(confusion_matrix)):
            report.write('{0}\t'.format(i+1))
        report.write('\n')
        for i in range(len(confusion_matrix)):
            report.write('{0}\t'.format(i+1))
            for j in range(len(confusion_matrix[i])):
                report.write('{0}\t'.format(confusion_matrix[i,j]))
            report.write('\n')
#    plt.plot(train_errors)
#    plt.plot(test_errors)
#    plt.legend(['train error','test error'], fontsize='small')
#    plt.title('Error Curve')
#    plt.savefig(report_path+'.png', dpi=450, bbox_inches='tight', pad_inches=0)
#    plt.clf()
    
#    if failure_number!=0 and failure_number<=50:
#        fialue_folder = os.path.join(os.path.dirname(report_path), 'failure_data')
#        os.makedirs(fialue_folder)
#        for i in range(failure_number):
#            plt.plot(failure_areas[i])
#            plt.legend(['Signal'], fontsize='small')
#            plt.title('classification failure signal')
#            plt.savefig(os.path.join(fialue_folder, str(i)+'.png'), dpi=450, bbox_inches='tight', pad_inches=0)
#            plt.clf()
#    plt.close('all')    
#read real data
data_normal = list(np.loadtxt('ECG_learning_Data_normal_3.txt', delimiter=','))
data_VPC = list(np.loadtxt('ECG_learning_Data_VPC.txt', delimiter=','))
data_STEMI = list(np.loadtxt('ECG_learning_Data_STEMI.txt', delimiter=','))
data_AF = list(np.loadtxt('ECG_learning_Data_AF.txt', delimiter=','))
data_close_TP_pairs = list(np.loadtxt('ECG_learning_Data_close_TP_pairs.txt', delimiter=','))
data_lawP = list(np.loadtxt('ECG_learning_Data_lawP.txt', delimiter=','))
data_artificial = list(np.loadtxt('ECG_learning_Data_artificial.txt', delimiter=','))
#data_noise = list(np.loadtxt('ECG_learning_Data_noise.txt', delimiter=','))

train_data_len = 1000
train_data_proportion = 0.5
train_step_length = 0.0001
#set maximun step
maximun_epoch = 2000

#set Test Model
Test_model = 'CNN_Classfication'


train_data, test_data = random_mix(train_data_proportion, data_normal, data_STEMI, data_artificial,
                                   data_AF, data_close_TP_pairs, data_lawP, data_VPC)
#train_data, test_data = random_mix(train_data_len, data_normal, data_STEMI, data_artificial,
#                                   data_AF, data_close_TP_pairs, data_lawP,data_VPC)


x_traindata = normalize(train_data[:,1:].astype('float32'))
y_traindata = np.zeros([len(x_traindata),8]).astype('float32')
for i in range(len(y_traindata)):
    y_traindata[i,train_data[i,0].astype('int')-1] = 1
    
x_traindata, y_traindata = random_batch(x_traindata, y_traindata, batch_num=500)

x_testdata = normalize(test_data[:,1:].astype('float32'))
y_testdata = np.zeros([len(x_testdata),8]).astype('float32')
for i in range(len(y_testdata)):
    y_testdata[i,test_data[i,0].astype('int')-1] = 1


sess = tf.InteractiveSession()

#x_data = np.array(range(28))[np.newaxis, : ]

x = tf.placeholder(tf.float32, [None, 800])
x_input = tf.reshape(x, [-1, 800, 1])

# conv layer 1
w_conv1 = weight_varible([25,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv1d(x_input, w_conv1) + b_conv1)
h_pool1 = max_pool_2(h_conv1)

# conv layer 2
w_conv2 = weight_varible([25,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv1d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2(h_conv2)

# full connection
W_fc1 = weight_varible([200 * 64, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 200 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## output layer: softmax
W_fc2 = weight_varible([256, 8])
b_fc2 = bias_variable([8])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 8])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(train_step_length).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())


train_errors = []
test_errors = []

start_time = time.time()
for i in range(maximun_epoch):
    batch_xs, batch_ys = random_batch(x_traindata, y_traindata, batch_num=100)
    train_error = sess.run(cross_entropy, feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 1.0})
    test_error = sess.run(cross_entropy, feed_dict={x:x_testdata,y_:y_testdata, keep_prob: 1.0})
    train_errors.append(train_error)
    test_errors.append(test_error)
    if (i+1) % 100 == 0 or i==0:
        # imformation on test   
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step {0:.3f}, training accuracy {1:.3f} %".format(i+1, train_accuracy))
        print('cross_entropy : {0}'.format(train_error))
        
        
    train_step.run(feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    
 
    
    epoch = i+1
#        print(sess.run(prediction, feed_dict={xs:batch_xs}))
#    if test_error <= 0.05:
#        epoch = i+1
#        break
#    if train_accuracy == 1:
#        epoch = i+1
#        break      
time_cost = time.time()-start_time
test_accuracy = accuracy.eval(feed_dict={x: x_testdata, y_: y_testdata, keep_prob: 1.0})
confusion_matrix, failue_index = confusion_matrix(x_testdata, y_testdata, 1.0)
submit_report(Test_model, y_traindata, y_testdata, time_cost, epoch, 
              train_step_length, test_accuracy, train_errors, test_errors, 
              confusion_matrix, x_testdata[failue_index])

# accuacy on test
print("final report :\n")
print('TRAINING MODEL    : {0}'.format(Test_model))
print('TRAINING STEP     : {0}'.format(train_step_length))        
print('TRAINING EPOCH    : {0}'.format(epoch))
print('TRAINING TIME     : {0:.3f} sec'.format(time_cost))
print('TEST ACCURACY     : {0:.3f}%'.format(test_accuracy*100))
print('CONFUSION MATRIX  : ')
print confusion_matrix
print('Mission complete')
sess.close()


#a = sess.run(y_conv, feed_dict={x:x_traindata, keep_prob: 1.0})
#b = sess.run(h_fc1, feed_dict={x:x_traindata})