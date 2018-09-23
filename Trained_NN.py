# Main file for NN model
import tensorflow as tf
import numpy as np
from enum import Enum
from data.create_data import Distribution
from functools import wraps

DATA_TYPE = Distribution.RANDOM


def set_data_type(data_type):
    global DATA_TYPE
    DATA_TYPE = data_type


# set parameter
class Parameter:
    def __init__(self, stages, cores, train_steps, batch_sizes, learning_rates, keep_ratios):
        self.stage_set = stages
        self.core_set = cores
        self.train_step_set = train_steps
        self.batch_size_set = batch_sizes
        self.learning_rate_set = learning_rates
        self.keep_ratio_set = keep_ratios


# parameter pool
class ParameterPool(Enum):
    RANDOM = Parameter(stages=[1, 5], cores=[[1, 32, 32, 1], [1, 8, 1]], train_steps=[30000, 30000],
                       batch_sizes=[128, 64], learning_rates=[0.0015, 0.0015], keep_ratios=[1.0, 1.0])
    LOGNORMAL = Parameter(stages=[1, 128], cores=[[1, 32, 32, 1], [1, 8, 1]], train_steps=[100000, 50000],
                          batch_sizes=[1024, 64], learning_rates=[0.001, 0.001], keep_ratios=[1.0, 0.9])
    EXPONENTIAL = Parameter(stages=[1, 100], cores=[[1, 8, 1], [1, 8, 1]], train_steps=[30000, 20000],
                            batch_sizes=[50, 50], learning_rates=[0.0001, 0.001], keep_ratios=[0.9, 1.0])
    EXPONENTIAL = Parameter(stages=[1, 100], cores=[[1, 16, 16, 1], [1, 8, 1]], train_steps=[20000, 300],
                            batch_sizes=[20, 50], learning_rates=[0.0001, 0.001], keep_ratios=[1.0, 1.0])
    NORMAL = Parameter(stages=[1, 100], cores=[[1, 8, 1], [1, 8, 1]], train_steps=[20000, 300],
                       batch_sizes=[50, 50], learning_rates=[0.0001, 0.001], keep_ratios=[0.9, 1.0])


# initialize weight
def weight_variable(shape):
    if DATA_TYPE == Distribution.RANDOM:
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        # initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.LOGNORMAL:
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        # initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.EXPONENTIAL:
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        # initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.NORMAL:
        initial = tf.truncated_normal(shape=shape, mean=0.1, stddev=0.1)
    else:
        initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# initialize bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Netural Network Model
class TrainedNN:
    def __init__(self, cores, train_step_num, batch_size, learning_rate, keep_ratio,
                 train_x, train_y, model_i, model_j):
        # set parameters
        if cores is None:
            cores = []
        self.core_nums = cores
        self.train_step_nums = train_step_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_ratio = keep_ratio
        self.train_x = train_x
        self.train_y = train_y
        self.model_i = model_i
        self.model_j = model_j
        self.train_flag = True
        self.sess = tf.Session()
        self.batch = 1
        self.batch_x = np.array([self.train_x[0:self.batch_size]]).T
        self.batch_y = np.array([self.train_y[0:self.batch_size]]).T
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.core_nums[-1]])
        self.w_fc = []
        self.b_fc = []
        for i in range(len(self.core_nums) - 1):
            self.w_fc.append(weight_variable(
                [self.core_nums[i], self.core_nums[i + 1]]))
            self.b_fc.append(bias_variable([self.core_nums[i + 1]]))
        self.h_fc = [None for i in range(len(self.core_nums))]
        self.h_fc_drop = [None for i in range(len(self.core_nums))]
        self.h_fc_drop[0] = tf.placeholder(
            tf.float32, shape=[None, self.core_nums[0]])
        self.keep_prob = tf.placeholder(tf.float32)

        # model structure
        for i in range(len(self.core_nums) - 2):
            self.h_fc[i] = tf.nn.leaky_relu(
                tf.matmul(self.h_fc_drop[i], self.w_fc[i]) + self.b_fc[i])
            self.h_fc_drop[i + 1] = tf.nn.dropout(self.h_fc[i], self.keep_prob)
        i = len(self.core_nums) - 2
        self.h_fc[i] = tf.matmul(
            self.h_fc_drop[i], self.w_fc[i]) + self.b_fc[i]

        # loss and optimizer
        self.mean_squared_error = tf.losses.mean_squared_error(
            self.y_, self.h_fc[len(self.core_nums) - 2])
        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.mean_squared_error)
        self.sess.run(tf.global_variables_initializer())

    # get next batch of data
    def next_batch(self):
        if self.batch * self.batch_size + self.batch_size < len(self.train_x):
            self.batch_x = np.array(
                [self.train_x[self.batch * self.batch_size:(self.batch + 1) * self.batch_size]]).T
            self.batch_y = np.array(
                [self.train_y[self.batch * self.batch_size:(self.batch + 1) * self.batch_size]]).T
            self.batch += 1
        else:
            self.batch_x = np.array(
                [self.train_x[self.batch * self.batch_size:len(self.train_x)]]).T
            self.batch_y = np.array(
                [self.train_y[self.batch * self.batch_size:len(self.train_y)]]).T
            self.batch = 0

    # train model
    def train(self):
        self.train_flag = False
        min_err = float("inf")
        for step in range(0, self.train_step_nums):
            self.sess.run(self.train_step, feed_dict={
                self.h_fc_drop[0]: self.batch_x, self.y_: self.batch_y,
                self.keep_prob: self.keep_ratio})
            # check every 1000 steps
            if step % 1000 == 0:
                err = self.sess.run(self.mean_squared_error, feed_dict={
                    self.h_fc_drop[0]: np.array([self.train_x]).T,
                    self.y_: np.array([self.train_y]).T,
                    self.keep_prob: 1.0})
                print("step: %d, mean_squared_error: %f" % (step, err))
                if err < min_err:
                    self.save("./model/model_" + str(self.model_i) +
                              "_" + str(self.model_j) + ".ckpt")
                    min_err = err
                    print("save model_" + str(self.model_i) +
                          "_" + str(self.model_j))
            self.next_batch()
        self.train_flag = False

    # predict
    def predict(self, feed_x=None):
        if feed_x is None:
            feed_x = self.train_x
        self.restore("./model/model_" + str(self.model_i) +
                     "_" + str(self.model_j) + ".ckpt")
        predicts = self.sess.run(self.h_fc[len(self.core_nums) - 2], feed_dict={
            self.h_fc_drop[0]: np.array([feed_x]).T, self.keep_prob: 1.0})
        return predicts

    # calculate standard error
    def cal_err(self):
        self.restore("./model/model_" + str(self.model_i) +
                     "_" + str(self.model_j) + ".ckpt")
        mse = self.sess.run(self.mean_squared_error, feed_dict={
            self.h_fc_drop[0]: np.array([self.train_x]).T,
            self.y_: np.array([self.train_y]).T, self.keep_prob: 1.0})
        std_err = np.sqrt(mse / len(self.train_y))
        return std_err

    # save model
    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    # restore model
    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
