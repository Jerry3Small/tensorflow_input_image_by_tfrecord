import os
from array import *
from random import shuffle

from tensorflow.examples.tutorials.mnist import input_data

from scipy import misc
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf

from conv_cards_utils import transform_to_3




import tensorflow as tf
# Create session
sess = tf.Session()

# 1. Define variables and initialize them, then construct the model (shell)

x = tf.placeholder(tf.float32, shape=[None, 784]) # input variable
y_ = tf.placeholder(tf.float32, shape=[None, 3])  # label variable (reality)

W = tf.Variable(tf.zeros([784, 3]))               # weight variable
b = tf.Variable(tf.zeros([3]))                    # bais

# 2. Initialize variables, construct model and define the evaluation metrics

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 3. Define the detail of neural network

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Note no initialize

saver = tf.train.Saver()
saver.restore(sess, "./cards_model_30000/cards_model.ckpt")

# Load from and save to
data_path = ['./jpg_to_mnist/training_images', './jpg_to_mnist/testing_images']
#data_path = ['./jpg_to_mnist/sample']

for set_path in data_path:

    data_image = array('B')
    data_label = array('B')

    FileList = []

    num_mst = 0
    num_spl = 0
    num_trp = 0

    for dirname in os.listdir(set_path): # dirname = label
        label_path = os.path.join(set_path,dirname)
        cnt = 0
        for filename in os.listdir(label_path):
            if filename.endswith(".png"):
                FileList.append(os.path.join(set_path,dirname,filename))
                cnt += 1
        if dirname == str(0):
            num_mst = cnt
        elif dirname == str(1):
            num_spl = cnt
        elif dirname == str(2):
            num_trp = cnt


    num_all = len(FileList)


    print (" * Processing " + set_path + " ...")

    print (" * #files: "  + str(num_all) +
           ", #mst: " + str(num_mst) +
           ", #spl: " + str(num_spl) + 
           ", #trp: " + str(num_trp))

    num_incorrect = 0
    num_mst_incrt = 0
    num_spl_incrt = 0
    num_trp_incrt = 0

    for filename in FileList:
        #print (" * " + filename)
        label = int(filename.split('/')[3])
        imgTarget = Image.open(filename).convert('L')
        imgTarget = imgTarget.resize((28, 28), PIL.Image.ANTIALIAS)
        imgTarget.save("../tmp_image.jpg")
        imgTarget = misc.imread("../tmp_image.jpg")
        imgTarget.shape=(1, 784)
        result = sess.run(y_conv, feed_dict={x: imgTarget, keep_prob: 1.0})
        predicted = int(sess.run(tf.argmax(result, 1)))

        if predicted != label:
            num_incorrect += 1
            if label == 0:
                num_mst_incrt += 1
            elif label == 1:
                num_spl_incrt += 1
            elif label == 2:
                num_trp_incrt += 1

    error_all = "{:.0%}".format(num_incorrect / num_all)
    error_mst = "{:.0%}".format(num_mst_incrt / num_mst)
    error_spl = "{:.0%}".format(num_spl_incrt / num_spl)
    error_trp = "{:.0%}".format(num_trp_incrt / num_trp)
    print (" * ALL: " + (error_all))
    print (" * MST: " + (error_mst))
    print (" * SPL: " + (error_spl))
    print (" * TRP: " + (error_trp))

