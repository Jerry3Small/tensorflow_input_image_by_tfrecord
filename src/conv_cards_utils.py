# PATH: /home/jerry3chang/Workspace/tensorflow_py3/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_module
# PATH: /home/jerry3chang/Workspace/tensorflow_py3/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

import math
import h5py
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

import os
from array import *
from random import shuffle

from PIL import Image

import numpy as np

# the generated mnist will still get one-hot of 10, needs to be transformed
def transform_to_3(images, labels):
    trans = np.zeros((images.shape[0], 3))
    c = 0
    for e in labels:
        d = 0
        for f in e:
            if d < 2:
                trans[c, d] = labels[c, d]
            elif labels[c, 0] == 0 and labels[c, 1] == 0:
                trans[c, 2] = 1
            else:
                trans[c, 2] = 0
            d += 1
        #if c <= 15:
            #print (e)
            #print (trans[c,:])
        c += 1
    return trans


# copy from mnist
def read_data_sets(train_dir,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   source_url=None): # omit url since we are using our own dataset
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 'test-images-idx3-ubyte.gz'
    TEST_LABELS = 'test-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir, None) # omit url, local file will be a path

    # type: DataSets
    with gfile.Open(local_file, 'rb') as f:
        train_images = mnist_module.extract_images(f)

    print (train_images.shape)

    local_file = base.maybe_download(TRAIN_LABELS, train_dir, None) # omit url
    with gfile.Open(local_file, 'rb') as f:
        train_labels = mnist_module.extract_labels(f, one_hot=one_hot)

    local_file = base.maybe_download(TEST_IMAGES, train_dir, None) # omit url
    with gfile.Open(local_file, 'rb') as f:
        test_images = mnist_module.extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, train_dir, None) # omit url
    with gfile.Open(local_file, 'rb') as f:
        test_labels = mnist_module.extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = mnist_module.DataSet(train_images, train_labels, **options)
    validation = mnist_module.DataSet(validation_images, validation_labels, **options)
    test = mnist_module.DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)




def load_dataset():
    train_dataset = h5py.File('./datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    # (1080,64,64,3)
    
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('./datasets/test_signs.h5', "r")

    print ("test_dataset.type " + str(type(test_dataset)))
    print ("test_dataset[\"test_set_x\"][:].type " + str(type(test_dataset["test_set_x"][:])))


    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    print ("test_set.type " + str(type(test_set_x_orig)))
    print ("test_set.ndim " + str(test_set_x_orig.ndim))
    print ("test_set.shape " + str(test_set_x_orig.shape))
    print ("test_set[0].type " + str(type(test_set_x_orig[0])))
    print ("test_set[0].ndim " + str(test_set_x_orig[0].ndim))
    print ("test_set[0].shape " + str(test_set_x_orig[0].shape))


    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_dataset_deck():
    # Load from and save to
    Names = [['./jpg_to_mnist/training_images','train'], ['./jpg_to_mnist/testing_images','test']]

    train_images = np.zeros(1620528) # 689 * 28 * 28 * 3
    train_labels = np.zeros(689)
    test_images = np.zeros(181104)
    test_labels = np.zeros(77)

    for name in Names:

        data_image = array('B')
        data_label = array('B')

        FileList = []

        # NOTE: [1:] Excludes .DS_Store from Mac OS
        for dirname in os.listdir(name[0]):
            path = os.path.join(name[0],dirname)
            for filename in os.listdir(path):
                if filename.endswith(".png"):
                    FileList.append(os.path.join(name[0],dirname,filename))

        print (str(name) + " " + str(len(FileList)))

        shuffle(FileList) # Usefull for further segmenting the validation set

        cnt = 0

        for filename in FileList:
            cnt += 1
            # NOTE: The class labels have to be integer
            label = int(filename.split('/')[3])

            im = Image.open(filename) # .convert('L')

            pixel = im.load()
            width, height = im.size # 28*28

            for channel in range(0,len(pixel[0,0])):
                for x in range(0,width):
                    for y in range(0,height):
                        data_image.append(pixel[y,x][channel])

            data_label.append(label) # labels start (one unsigned byte each)

        if name[1] == 'train':
            train_images = np.array(data_image).reshape(len(FileList), 28, 28, 3)
            train_labels = np.array(data_label).reshape(len(FileList))
        elif name[1] == 'test':
            test_images = np.array(data_image).reshape(len(FileList), 28, 28, 3)
            test_labels = np.array(data_label).reshape(len(FileList))

    return train_images, train_labels, test_images, test_labels
