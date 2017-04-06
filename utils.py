from glob import glob
import os
import numpy as np
import ujson as json
from matplotlib import pyplot as plt
from datetime import datetime
import bcolz

import pandas as pd
import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice, randint
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Activation
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.preprocessing import image, sequence
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

	
def get_batches(dirname, gen = image.ImageDataGenerator(), shuffle = True, batch_size = 4, class_mode = 'categorical', target_size = (224,224)):
	return gen.flow_from_directory(dirname, target_size=target_size, class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
	
def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])
	
def plot_log_loss(loss_list, val_loss_list, title = 'log10(model loss)'):
	plt.plot(np.log10(loss_list))
	plt.plot(np.log10(val_loss_list))
	plt.title(title)
	plt.ylabel('log10(loss)')
	plt.xlabel('epoch')
	plt.legend(['train', 'valid'], loc='upper left')
	plt.show()

def do_clip(arr, mx): 
	return np.clip(arr, (1-mx)/7, mx)

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


