# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 01:37:04 2016

@author: siang
"""
import MyecgError
import tensorflow as tf
import numpy as np

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


def normalize(array):
    arrmin = np.min(array)
    array = array-arrmin
    arrmax = np.max(array)
    return array/arrmax


#def random_batch(example, label, batch_num=None):
#    min_after_dequeue = 10000
#    capacity = min_after_dequeue + 3 * 100
#    batch_xs, batch_ys = tf.train.shuffle_batch([x_traindata, y_train_data], 
#                                batch_size=100, capacity=capacity,
#                                min_after_dequeue=min_after_dequeue)
#    with tf.Session() as sess:
#        a = batch_xs.eval()
#        b = batch_ys.eval())                          
#    return batch_xs, batch_ys
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

def random_sampling(amount, *args):
    for count,thing in enumerate(arg):
        
#def  input_pipeline():
    
    
#read real data
data_normal = list(np.loadtxt('ECG_learning_Data_normal.txt', delimiter=','))
data_STEMI = list(np.loadtxt('ECG_learning_Data_STEMI.txt', delimiter=','))
data_combined = list(np.loadtxt('ECG_learning_Data_combined.txt', delimiter=','))
data_AF = list(np.loadtxt('ECG_learning_Data_AF.txt', delimiter=','))
data_close_TP_pairs = list(np.loadtxt('ECG_learning_Data_close_TP_pairs.txt', delimiter=','))
data_lawP = list(np.loadtxt('ECG_learning_Data_lawP.txt', delimiter=','))
#data_noise = list(np.loadtxt('ECG_learning_Data_noise.txt', delimiter=','))


#train_data = np.array(data_normal[:150] + data_STEMI[:37])
#test_data = np.array(data_normal[150:] + data_STEMI[37:])
train_data = np.array(data_STEMI[:37]  + data_STEMI[:37] + data_STEMI[:37] + 
        data_STEMI[:37] + data_normal[:150] + data_combined[:150] + data_AF[:50] + 
        data_AF[:50] + data_AF[:50] + data_close_TP_pairs[:150] + data_lawP[:50] + 
        data_lawP[:50] + data_lawP[:50])

#test_data = np.array(data_STEMI[37:] + data_normal[150:] + data_combined[150:300])
#train_data = np.array(data_STEMI[:37] + data_STEMI[:37] + data_STEMI[:37] + 
#                        data_STEMI[:37] + data_normal[:150] + data_combined[:150])
test_data = np.array(data_STEMI[37:]+ data_normal[150:] + data_combined[150:] + 
                    data_close_TP_pairs[150:] + data_AF[50:] + data_lawP[50:])

x_traindata = normalize(train_data[:,1:].astype('float32'))
y_traindata = np.zeros([len(x_traindata),6]).astype('float32')
for i in range(len(y_traindata)):
    y_traindata[i,train_data[i,0].astype('int')-1] = 1
    
x_traindata, y_traindata = random_batch(x_traindata, y_traindata, batch_num=500)

x_testdata = normalize(test_data[:,1:].astype('float32'))
y_testdata = np.zeros([len(x_testdata),6]).astype('float32')
for i in range(len(y_testdata)):
    y_testdata[i,test_data[i,0].astype('int')-1] = 1

# Make up some real data
#x_data = np.linspace(-1,1,300)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, x_data.shape)
#y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('Input') as scope:
    xs = tf.placeholder(tf.float32, [None, 800], name='X_input')
    ys = tf.placeholder(tf.float32, [None, 6], name= 'Y_input')
# add hidden layer
    
#hidden_1 = add_layer('hidden1', xs, 800, 64, activation_function = tf.nn.softmax)
#hidden_2 = add_layer('hidden2', hidden_1, 64, 12, activation_function = tf.nn.softmax)   
    
prediction = add_layer('output', xs, 800, 6, activation_function = tf.nn.softmax)

# the error between prediciton and real data
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                     reduction_indices=[1]))
with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    tf.scalar_summary('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

# important step
init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('logs/', sess.graph)
sess.run(init)

for i in range(1000):
    # training
#    batch_xs = x_traindata
#    batch_ys = y_train_data
    batch_xs, batch_ys = random_batch(x_traindata, y_traindata, batch_num=100)
    sess.run(train_step, feed_dict={xs:batch_xs,ys:batch_ys})
    if (i+1) % 10 == 0:
#        result = sess.run(merged, feed_dict={xs:batch_xs,ys:batch_ys})
#        writer.add_summary(result, i)
        accuracy = compute_accuracy(x_testdata, y_testdata)
        print('step {0}\naccuracy : {1:.3f}%'.format(i+1,accuracy*100))
        print('cross_entropy : {0}'.format(sess.run(cross_entropy, feed_dict={xs:batch_xs,ys:batch_ys})))
    #        print(sess.run(prediction, feed_dict={xs:batch_xs}))
        if accuracy==1:
            break
print('Mission complete')
sess.close()