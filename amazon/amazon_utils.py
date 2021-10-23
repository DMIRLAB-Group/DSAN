from __future__ import print_function

from sklearn.datasets import load_svmlight_files
import numpy as np
import tensorflow as tf


def load_amazon(source_name, target_name, data_folder=None, verbose=False):
    if data_folder is None:
        data_folder = './data/'
    source_file = data_folder + source_name + '_train.svmlight'
    target_file = data_folder + target_name + '_train.svmlight'
    test_file = data_folder + target_name + '_test.svmlight'
    if verbose:
        print('source file:', source_file)
        print('target file:', target_file)
        print('test file:  ', test_file)

    xs, ys, xt, yt, xt_test, yt_test = load_svmlight_files([source_file, target_file, test_file])
    ys, yt, yt_test = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, yt_test))

    return xs.toarray(), ys, xt.toarray(), yt, xt_test.toarray(), yt_test


def csr_2_sparse_tensor_tuple(csr_matrix):
    coo_matrix = csr_matrix.tocoo()
    indices = np.transpose(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = coo_matrix.data
    shape = csr_matrix.shape
    return indices, values, shape


def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def conv2d(x, W, padding="SAME"):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


if __name__ == '__main__':
    source_train_input, source_train_label, \
    target_train_input, target_train_label, \
    target_test_input, target_test_label = load_amazon(
        source_name="books", target_name="books", data_folder="dataset/data/")

    print(len(target_test_label))

    target_count = 0
    for i in target_test_label:
        if i == 0:
            target_count += 1

    print(target_count)