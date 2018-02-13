from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

from scipy import misc
import PIL
from PIL import Image
import numpy as np

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.zeros([784, 3]))
b = tf.Variable(tf.zeros([3]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
    # next_batch() will returns a tuple of two arrays
    batch = mnist.train.next_batch(100)
    trans = np.zeros((batch[0].shape[0], 3))
    #print (batch[0].shape) # (100, 784)
    #print (batch[1].shape) # (100, 10)
    #print (batch[1][0,:])
    #print (batch[1][1,:])
    #print (len(batch[1][1,:])) # 10
    #print (len(batch[1])) # 100
    #print ("iteration: " + str(i))
    c = 0
    for e in batch[1]:
        d = 0
        for f in e:
            if d < 2:
                trans[c, d] = batch[1][c, d]
            elif batch[1][c, 0] == 0 and batch[1][c, 1] == 0:
                trans[c, 2] = 1
            else:
                trans[c, 2] = 0
            d += 1
        #if c <= 15:
            #print (e)
            #print (trans[c,:])
        c += 1
    train_step.run(feed_dict={x: batch[0], y_: trans})

max_prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print (type(mnist))                   # <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>
#print (mnist.train.num_examples)      # 55000
#print (mnist.validation.num_examples) # 5000
#print (mnist.test.num_examples)       # 10000
#print (type(mnist.test))
#print (type(mnist.test.images))
#print (type(mnist.test.images[0]))

#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#print(y.eval(feed_dict={x: [mnist.test.images[0], mnist.test.images[1]], y_: [mnist.test.labels[0], mnist.test.labels[1]]}))
#print(max_prediction.eval(feed_dict={x: [mnist.test.images[0], mnist.test.images[1]], y_: [mnist.test.labels[0], mnist.test.labels[1]]}))
#print(mnist.test.labels[0])
#print(mnist.test.labels[1])


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

img = misc.imread("../tmp_image.jpg")
img.shape=(1, 784)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()




# TRAIN - BEGIN
with tf.device("/gpu:0"):
    for i in range(5000): # 50000
        batch = mnist.train.next_batch(50)

        trans = np.zeros((batch[0].shape[0], 3))
        c = 0
        for e in batch[1]:
            d = 0
            for f in e:
                if d < 2:
                    trans[c, d] = batch[1][c, d]
                elif batch[1][c, 0] == 0 and batch[1][c, 1] == 0:
                    trans[c, 2] = 1
                else:
                    trans[c, 2] = 0
                d += 1
            c += 1

        if i%500 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: trans, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            save_path = saver.save(sess, "/home/jerry3chang/Desktop/tf_input_jpg/model_jerry.ckpt")
            print("model saved in file: %s" %save_path)

        train_step.run(feed_dict={x: batch[0], y_: trans, keep_prob: 0.5})

# print (mnist.test.images.shape) # (10000, 784)
# print (mnist.test.labels.shape) # (10000, 10)

trans = np.zeros((mnist.test.images.shape[0], 3))
c = 0
for e in mnist.test.labels:
    d = 0
    for f in e:
        if d < 2:
            trans[c, d] = mnist.test.labels[c, d]
        elif mnist.test.labels[c, 0] == 0 and mnist.test.labels[c, 1] == 0:
            trans[c, 2] = 1
        else:
            trans[c, 2] = 0
        d += 1
    c += 1

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: trans, keep_prob: 1.0}))
save_path = saver.save(sess, "/home/jerry3chang/Desktop/tf_input_jpg/model_jerry.ckpt")
print("model saved in file: %s" %save_path)
# TRAIN - END


print("PROCESSING INPUT ...")

flags = tf.app.flags
FLAGS = flags.FLAGS

# NOTE the input must be fixed to 28*28
# predict num0_0.jpg will fail
flags.DEFINE_string("image_path", "../num0_0.jpg", "Path to your input digit image.")

imgTarget = Image.open(FLAGS.image_path)
print("Orignial image size is: ")
print(imgTarget.size)
imgTarget = imgTarget.resize((28, 28), PIL.Image.ANTIALIAS)
imgTarget.save("../tmp_image.jpg")
print("Resized image size is: ")
print(imgTarget.size)

imgTarget = misc.imread("../tmp_image.jpg")
imgTarget.shape=(1, 784)

result = sess.run(y_conv, feed_dict={x: imgTarget, keep_prob: 1.0})
print("The output of the network is: ")
print(result)
print("Prediction is (output one-hot digit from 0 to 9):")
print(sess.run(tf.argmax(result, 1)))


