import tensorflow as tf
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import matplotlib.pyplot as plt

from PIL import Image
from constant import *


# set the weights to be small random values, with truncated normal distribution
def weight_variable(shape, myname, train):
    """
    Create weight variable
    :param shape:
    :param myname:
    :param train:
    :return:
    """
    initial = tf.random_uniform(shape, minval=-0.1, maxval = 0.1)
    return tf.Variable(initial, name=myname, trainable=train)

def draw_bbox(image_path, locations):
    locations = toMnistCoordinates(locations)
    image_path = np.squeeze(np.uint8(image_path), -1)
    im = Image.fromarray(np.uint8(image_path), "L")
    im = im.convert("RGB")
    try:
        font = ImageFont.truetype('arial.ttf', 3)
    except IOError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(im)
    colors = ['red', 'yellow', 'blue', 'green', 'pink', 'purple']
    for index, location in enumerate(locations, 1):
        # text_bottom = location[0] + 6 + 3
        left = location[1] - 6
        right = location[1] + 6
        top = location[0] + 6
        bottom = location[0] - 6
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=1, fill=colors[index-1])
    return im


def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    image_with_boxes = tf.py_func(draw_bbox, image_and_detections,
                                  tf.uint8)
    return image_with_boxes


def affineTransform(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w = tf.get_variable("w", [x.get_shape()[1], output_dim])
    b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, w)+b


def toMnistCoordinates(coordinate_tanh):
    '''
    Transform coordinate in [-1,1] to mnist
    :param coordinate_tanh: vector in [-1,1] x [-1,1]
    :return: vector in the corresponding mnist coordinate
    '''
    return np.round(((coordinate_tanh + 1) / 2.0) * img_size)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('param_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('param_mean/' + name, mean)
        with tf.name_scope('param_stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('param_sttdev/' + name, stddev)
        tf.summary.scalar('param_max/' + name, tf.reduce_max(var))
        tf.summary.scalar('param_min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def plotWholeImg(img, img_size, sampled_locs_fetched):
    """

    :param img:
    :param img_size:
    :param sampled_locs_fetched:
    :return:
    """
    plt.imshow(np.reshape(img, [img_size, img_size]),
               cmap=plt.get_cmap('gray'), interpolation="nearest")

    plt.ylim((img_size - 1, 0))
    plt.xlim((0, img_size - 1))

    # transform the coordinate to mnist map
    sampled_locs_mnist_fetched = toMnistCoordinates(sampled_locs_fetched)
    # visualize the trace of successive nGlimpses (note that x and y coordinates are "flipped")
    plt.plot(sampled_locs_mnist_fetched[0, :, 1], sampled_locs_mnist_fetched[0, :, 0], '-o',
             color='lawngreen')
    plt.plot(sampled_locs_mnist_fetched[0, -1, 1], sampled_locs_mnist_fetched[0, -1, 0], 'o',
             color='red')

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# to use for maximum likelihood with input location
def gaussian_pdf(mean, sample):
    """

    :param mean:
    :param sample:
    :return:
    """
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
    return Z * tf.exp(a)