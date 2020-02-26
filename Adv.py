
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd
from scipy.misc import imread
from scipy.misc import imsave, imresize
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate

import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

tf.flags.DEFINE_string('checkpoint_path',
                       '',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 32.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 16, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 32, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

FLAGS = tf.flags.FLAGS


model_checkpoint_map = {
    'vgg': os.path.join(FLAGS.checkpoint_path, 'vgg_16.ckpt')}


def load_images_with_labels(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    labels = []
    vgg_labels = []
    idx = 0
    batch_size = batch_shape[0]

    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['ImageId']: dev.iloc[i]['TrueLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():

        image = imread(os.path.join(input_dir, filename), mode='RGB')
        image = imresize(image, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
        image[:, :, 0] -= _R_MEAN
        image[:, :, 1] -= _G_MEAN
        image[:, :, 2] -= _B_MEAN
        images[idx, :, :, :] = image
        filenames.append(filename)
        labels.append(filename2label[filename])
        vgg_labels.append(filename2label[filename] - 1)
        idx += 1
        if idx == batch_size:
            yield filenames, images, labels, vgg_labels
            filenames = []
            labels = []
            vgg_labels = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images, labels, vgg_labels


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.
    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):

        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            image = images[i, :, :, :]
            image[:, :, 0] += _R_MEAN
            image[:, :, 1] += _G_MEAN
            image[:, :, 2] += _B_MEAN
            image = imresize(image, [299, 299])
            imsave(f, image, format='png')


def image_augmentation(x):
    # img, noise
    one = tf.fill([tf.shape(x)[0], 1], 1.)
    zero = tf.fill([tf.shape(x)[0], 1], 0.)
    transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
    rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=0.05), zero, zero], axis=1)
    return images_transform(x, transforms + rands, interpolation='BILINEAR')


def image_rotation(x):
    """ imgs, scale, scale is in radians """
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)
    return images_rotate(x, rands, interpolation='BILINEAR')


def input_diversity(input_tensor):
    input_tensor_ = image_augmentation(input_tensor)
    input_tensor__ = image_rotation(input_tensor_)
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor__, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret


def graph(x, y, i, x_max, x_min, grad, vgg_y):
    eps = FLAGS.max_epsilon
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum

    one_hot_vgg = tf.one_hot(vgg_y, 1000)

    x_div = input_diversity(x)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(x_div, num_classes=1001, is_training=False)

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_vgg, logits)

    noise = tf.gradients(cross_entropy, x)[0]

    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise, vgg_y


def stop(x, y, i, x_max, x_min, grad, vgg_y):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    start_time = time.time()

    eps = FLAGS.max_epsilon

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = x_input + eps
        x_min = x_input - eps
        y = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        # y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        vgg_y = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad, vgg_y])

        # Run computation
        s = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

        print('Building Graph Done', time.time() - start_time)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config) as sess:
            s.restore(sess, model_checkpoint_map['vgg'])

            print('Load Parameters Done', time.time() - start_time)
            tot_images = 0

            for filenames, images, labels, vgg_labels in load_images_with_labels(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images, y: labels, vgg_y: vgg_labels})
                save_images(adv_images, filenames, FLAGS.output_dir)

                tot_images += len(filenames)
                print(tot_images, time.time() - start_time)

    print((time.time() - start_time)/60)


if __name__ == '__main__':
    tf.app.run()
