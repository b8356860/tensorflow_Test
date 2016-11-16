# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 01:37:04 2016

@author: siang
"""
import MyecgError
import tensorflow as tf
import numpy as np
import sklearn as sk
import os
import time
import platform
import matplotlib.pyplot as plt
def add_layer(name, inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope(name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_uniform([in_size, out_size]), name='W')
            tf.histogram_summary(name + '/weights', Weights)
        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='B')
            tf.histogram_summary(name + '/biases', biases)
        with tf.name_scope('Wx_plus_B'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.histogram_summary(name + '/outputs', outputs)
        return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


def confusion_matrix(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    result = tf.argmax(y_pre, 1)
    true = tf.argmax(v_ys, 1)
    y_result , y_true= sess.run([result,true], feed_dict={xs:v_xs, ys:v_ys})
    failue_index = np.where((y_result!=y_true))
    conf_matrix = sk.metrics.confusion_matrix(y_true, y_result)
    return conf_matrix, failue_index


def normalize(array):
    arrmin = np.min(array)
    array = array-arrmin
    arrmax = np.max(array)
    return array/arrmax


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
    report_path = os.path.join('reports', model, 'report'+number,'report'+number)
    os.makedirs(os.path.dirname(report_path))
    system_inform = platform.uname()
    failure_number = len(failure_areas)
    with open(report_path+'.txt', 'w') as report:
        report.write('REPORT NO.{0} {1}\n'.format(number, system_inform))
        report.write('TRAINING MODEL    : {0}\n'.format(model))
        report.write('TRAINING STEP     : {0}\n'.format(step))        
        report.write('TRAINING EPOCH    : {0}\n'.format(epoch))
        report.write('TRAINING TIME     : {0:.3f} sec\n'.format(time_cost))
        report.write('TEST ACCURACY     : {0:.3f}%\n'.format(accuracy))
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
    plt.plot(train_errors)
    plt.plot(test_errors)
    plt.legend(['train error','test error'], fontsize='small')
    plt.title('Error Curve')
    plt.savefig(report_path+'.png', dpi=450, bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    if failure_number!=0 and failure_number<=50:
        fialue_folder = os.path.join(os.path.dirname(report_path), 'failure_data')
        os.makedirs(fialue_folder)
        for i in range(failure_number):
            plt.plot(failure_areas[i])
            plt.legend(['Signal'], fontsize='small')
            plt.title('classification failure signal')
            plt.savefig(os.path.join(fialue_folder, str(i)+'.png'), dpi=450, bbox_inches='tight', pad_inches=0)
            plt.clf()
    plt.close('all')


def add_noise(data):
    for i in range(len(data)):
        data[i,1:] + np.random.random(len(data[i])-1)
    return data

data_list = {'normal','artificial','STEMI','close_TP_pairs','close_TP_pairs','lawP'}    
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
train_step_length = 0.2
#set maximun step
maximun_epoch = 3000

#set Test Model
#Test_model = 'Basic_Classfication'
#Test_model = 'Basic_Classfication_unbalanced'
#Test_model = 'Basic_Classfication_balanced'
Test_model = 'Basic_Classfication_balanced1000'
#Test_model = 'Basic_Classfication_balanced_test_3layers'

#add test data
#data_artificial[-1][0]=1
#data_lawP[-1][0] = 2
#data_STEMI[-1][0] = 5
#
#data_artificial[0][0]=1
#data_lawP[0][0] = 2
#data_STEMI[0][0] = 5

#train_data, test_data = random_mix(train_data_proportion, data_normal, data_STEMI, data_artificial,
#                                   data_AF, data_close_TP_pairs, data_lawP,data_VPC)
train_data, test_data = random_mix(train_data_len, data_normal, data_STEMI, data_artificial,
                                   data_AF, data_close_TP_pairs, data_lawP,data_VPC)

#train_data = np.array(data_normal[:150] + data_STEMI[:37])
#test_data = np.array(data_normal[150:] + data_STEMI[37:])


#train_data = np.array(data_STEMI[:37] + data_normal[:361] + data_artificial[:400]
#                 + data_AF[:56] + data_close_TP_pairs[:550] + data_lawP[:62]+data_VPC[:25])
#
#test_data = np.array(data_STEMI[37:] + data_normal[361:] + data_artificial[400:]
#                    + data_AF[56:] + data_close_TP_pairs[550:] + data_lawP[62:]+ data_VPC[:25])

#train_data = np.array(data_STEMI[:37] + data_STEMI[:37] + data_STEMI[:37] + 
#                        data_STEMI[:37] + data_normal[:150] + data_combined[:150])
#test_data = np.array(data_STEMI[37:]+ data_normal[150:] + data_combined[150:] + 
#                    data_close_TP_pairs[150:] + data_AF[50:] + data_lawP[50:])

x_traindata = normalize(train_data[:,1:].astype('float32'))
y_traindata = np.zeros([len(x_traindata),8]).astype('float32')
for i in range(len(y_traindata)):
    y_traindata[i,train_data[i,0].astype('int')-1] = 1
    
x_traindata, y_traindata = random_batch(x_traindata, y_traindata, batch_num=500)

x_testdata = normalize(test_data[:,1:].astype('float32'))
y_testdata = np.zeros([len(x_testdata),8]).astype('float32')
for i in range(len(y_testdata)):
    y_testdata[i,test_data[i,0].astype('int')-1] = 1

# Make up some real data
#x_data = np.linspace(-1,1,300)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, x_data.shape)
#y_data = np.square(x_data) - 0.5 + noise

#start time
start_time = time.time()

# define placeholder for inputs to network
with tf.name_scope('Input') as scope:
    xs = tf.placeholder(tf.float32, [None, 800], name='X_input')
    ys = tf.placeholder(tf.float32, [None, 8], name= 'Y_input')
# add hidden layer
    
#hidden_1 = add_layer('hidden1', xs, 800, 256, activation_function = tf.nn.softmax)
#hidden_2 = add_layer('hidden2', hidden_1, 256, 64, activation_function = None)   
    
prediction = add_layer('output', xs, 800, 8, activation_function = tf.nn.softmax)

# the error between prediciton and real data
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                     reduction_indices=[1]))
with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
#    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, ys))
    tf.scalar_summary('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(train_step_length).minimize(cross_entropy)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
#merged = tf.merge_all_summaries()
#writer = tf.train.SummaryWriter('logs/', sess.graph)
sess.run(init)

train_errors = []
test_errors = []
#accuracys = []
step = 0
for i in range(maximun_epoch):
    # training
#    batch_xs = x_traindata
#    batch_ys = y_train_data
    batch_xs, batch_ys = random_batch(x_traindata, y_traindata, batch_num=100)
    sess.run(train_step, feed_dict={xs:batch_xs,ys:batch_ys})
#    if (i+1) % 10 == 0:
#        result = sess.run(merged, feed_dict={xs:batch_xs,ys:batch_ys})
#        writer.add_summary(result, i)
    train_error = sess.run(cross_entropy, feed_dict={xs:batch_xs,ys:batch_ys})
    test_error = sess.run(cross_entropy, feed_dict={xs:x_testdata,ys:y_testdata})
    accuracy = compute_accuracy(x_testdata, y_testdata)
    train_errors.append(train_error)
    test_errors.append(test_error)
#        accuracys.append(accuracy)
    print('step {0}\naccuracy : {1:.3f}%'.format(i+1,accuracy*100))
    print('cross_entropy : {0}'.format(test_error))
    epoch = i+1
#        print(sess.run(prediction, feed_dict={xs:batch_xs}))
    if test_error <= 0.05:
        epoch = i+1
        break
    if accuracy == 1:
        epoch = i+1
        break       
    
time_cost = time.time()-start_time
confusion_matrix, failue_index = confusion_matrix(x_testdata, y_testdata)
submit_report(Test_model, y_traindata, y_testdata, time_cost, epoch, 
              train_step_length, accuracy*100, train_errors, test_errors, 
              confusion_matrix, x_testdata[failue_index])
print confusion_matrix
print('Mission complete')
sess.close()


#if __name__=='__main__':
#    for i in range(10):
#    main()