import tensorflow as tf
import tf_mnist_loader
import numpy as np
import sys
import os

from utils import *
from gen_data import *
from constant import *


dataset = tf_mnist_loader.read_data_sets("mnist_data")
save_dir = "chckPts/"
save_prefix = "save"
summaryFolderName = "summary/"

if len(sys.argv) == 2:
    simulationName = str(sys.argv[1])
    print("Simulation name = " + simulationName)
    summaryFolderName = summaryFolderName + simulationName + "/"
    save_dir= save_dir + simulationName + '/'
    imgsFolderName = "imgs/" + simulationName + "/"
    if os.path.isdir(summaryFolderName) == False:
        os.mkdir(summaryFolderName)

scales = ['0.5', '0.75', '1', '1.5', '2']
load_paths = []
for scale in scales:
    load_paths += ['./chckPts/{}_{}_c'.format('eccen', scale)]

# implements the input network
def glimpseSensor(img, normLoc):
    """
    Get Glimpse for the given location
    :param img:
    :param normLoc:
    :return:
    """
    loc = tf.round(((normLoc + 1) / 2.0) * img_size)  # normLoc coordinates are between -1 and 1
    loc = tf.cast(loc, tf.int32)

    img = tf.reshape(img, (batch_size, img_size, img_size, channels))

    # process each image individually
    zooms = []
    for k in range(batch_size):
        imgZooms = []
        one_img = img[k,:,:,:]
        max_radius = minRadius * (2 ** (depth - 1))
        offset = 2 * max_radius

        # pad image with zeros
        one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
                                               max_radius * 4 + img_size, max_radius * 4 + img_size)

        for i in range(depth):
            r = int(minRadius * (2 ** (i)))

            d_raw = 2 * r
            d = tf.constant(d_raw, shape=[1])
            d = tf.tile(d, [2])
            loc_k = loc[k,:]
            adjusted_loc = offset + loc_k - r
            one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value, one_img.get_shape()[1].value))

            # crop image to (d x d)
            zoom = tf.slice(one_img2, adjusted_loc, d)

            # resize cropped image to (sensorBandwidth x sensorBandwidth)
            zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (sensorBandwidth, sensorBandwidth))
            zoom = tf.reshape(zoom, (sensorBandwidth, sensorBandwidth))
            imgZooms.append(zoom)

        zooms.append(tf.stack(imgZooms))

    zooms = tf.stack(zooms)

    glimpse_images.append(zooms)

    return zooms


def get_glimpse_eccen(loc):
    """

    :param loc:
    :return:
    """
    # get input using the previous location
    glimpse_input = glimpseSensor(inputs_placeholder, loc)
    glimpse_input = tf.expand_dims(glimpse_input, 4)
    glimpse_input = tf.reshape(glimpse_input, (batch_size, 3, sensorBandwidth, sensorBandwidth, 1))
    # the hidden units that process location & the input
    ###act_glimpse_hidden = tf.nn.relu(tf.matmul(glimpse_input, Wg_g_h) + Bg_g_h)
    filters = weight_variable([1, 3, 3, 1, 32], 'conv3d_w', True)
    conv3d = tf.nn.conv3d(glimpse_input, filters, strides=[1, 1, 1, 1, 1], padding='VALID') # 20*3*8*8*32
    conv3d = tf.nn.relu(conv3d)
    conv3d = tf.nn.max_pool3d(conv3d, [1,1,2,2,1], [1,1,2,2,1], padding="VALID")

    filters2 = weight_variable([1, 2, 2, 32, 32], 'conv3d_w2', True)
    conv3d = tf.nn.conv3d(conv3d, filters2, strides=[1,1,1,1,1], padding='VALID')
    conv3d = tf.nn.relu(conv3d)
    conv3d = tf.nn.max_pool3d(conv3d, [1,3,2,2,1], [1,3,2,2,1], padding="VALID")

    conv3d_reshape = tf.reshape(conv3d, (batch_size, 128))
    # act_glimpse_hidden = tf.nn.relu(conv3d_reshape + weight_variable((1, 128), 'conv3d_b', True))
    glimpseFeature1 = tf.matmul(conv3d_reshape, weight_variable((128, n_classes), 'fc_w', True)
                                           + weight_variable((1,n_classes), 'fc_b', True))

    return glimpseFeature1


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, FLAGS.NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def evaluate():
    data = dataset.test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0

    for i in range(batches_in_epoch):
        nextX, nextY = dataset.test.next_batch(batch_size)
        if translateMnist:
            nextX, _ = convertTranslated(nextX, MNIST_SIZE, MNIST_SIZE, img_size)
        feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}
        correct_num = sess.run(correct, feed_dict=feed_dict)
        accuracy += np.sum(correct_num)

    accuracy /= len(data._images)
    print(("ACCURACY: " + str(accuracy)))


def evaluate_only(scale_size):
    data = dataset.test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0

    for i in range(batches_in_epoch):
        nextX, nextY = dataset.test.next_batch(batch_size)
        if translateMnist:
            nextX, _ = convertTranslated(nextX, MNIST_SIZE, scale_size, img_size)
        feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}
        y_pred = sess.run(prediction, feed_dict=feed_dict)
        y_pred = tf.nn.softmax(y_pred, name="output")
        y_pred = tf.argmax(y_pred, 1)
        correct = tf.equal(y_pred, nextY)
        accuracy += correct

    accuracy /= batches_in_epoch
    print(("{} ACCURACY: ".format(scale_size) + str(accuracy)))


def evaluate_cluttered(trans_size):
    """

    :param trans_size:
    :return:
    """
    data = dataset.test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0

    for i in range(batches_in_epoch):
        nextX, nextY = dataset.test.next_batch(batch_size)
        if translateMnist:
            nextX, _ = convertCluttered(nextX, MNIST_SIZE, trans_size, img_size)
        feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}
        y_pred = sess.run(prediction, feed_dict=feed_dict)
        y_pred = tf.nn.softmax(y_pred, name="output")
        y_pred = tf.argmax(y_pred, 1)
        correct = tf.equal(y_pred, nextY)
        accuracy += correct

    accuracy /= batches_in_epoch
    print(("Cluttered {} ACCURACY: ".format(trans_size) + str(accuracy)))


with tf.Graph().as_default():
    # set the learning rate
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(initLr, global_step, lrDecayFreq, lrDecayRate, staircase=True)

    # preallocate x, y, baseline
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size), name="labels_raw")
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_size * img_size), name="images")

    initial_loc = tf.zeros((batch_size, 2))
    prediction = get_glimpse_eccen(initial_loc)
    y_pred = tf.nn.softmax(prediction, name="output")
    y_pred = tf.argmax(y_pred, 1)
    correct = tf.cast(tf.equal(y_pred, labels_placeholder), tf.float32)

    loss = loss(prediction, labels_placeholder)
    optimizer = tf.train.AdagradOptimizer(lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)

    if eval_only:
        for path in load_paths:
            ckpt_path = tf.train.latest_checkpoint(path)
            saver.restore(sess, ckpt_path)
            print(ckpt_path)
            evaluate_only(14)
            evaluate_only(21)
            evaluate_only(28)
            evaluate_only(42)
            evaluate_only(56)
            evaluate_cluttered(14)
            evaluate_cluttered(21)
            evaluate_cluttered(28)
            evaluate_cluttered(42)
            evaluate_cluttered(56)
    else:
        for epoch in range(start_step + 1, max_iters):
            # get the next batch of examples
            nextX, nextY = dataset.train.next_batch(batch_size)
            if translateMnist:
                if MixedMnist:
                    list_scales = [int(0.5 * MNIST_SIZE), int(0.75 * MNIST_SIZE), MNIST_SIZE, int(1.5 * MNIST_SIZE),
                                   int(2 * MNIST_SIZE)]
                    if clutteredMnist:
                        nextX, nextX_coord = convertCluttered_mix(nextX, MNIST_SIZE, list_scales, img_size)
                    else:
                        nextX, nextX_coord = convertTranslated_mix(nextX, MNIST_SIZE, list_scales, img_size)
                else:
                    if clutteredMnist:
                        nextX, nextX_coord = convertCluttered(nextX, MNIST_SIZE, translateMnist_scale, img_size)
                    else:
                        nextX, nextX_coord = convertTranslated(nextX, MNIST_SIZE, translateMnist_scale, img_size)
            feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}
            _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)

            if epoch % 50 == 0:
                print(('Step %d: cost = %.5f'% (epoch, batch_loss)))

                if epoch % 2000 == 0:
                    saver.save(sess, save_dir + save_prefix + str(epoch) + ".ckpt")
                    evaluate()

    sess.close()



