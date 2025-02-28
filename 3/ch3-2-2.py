import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20171213)
tf.set_random_seed(20160614)

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

num_units = 1024

class SingleLayerNetwork :
    def __init__(self, num_units):
        with tf.Graph().as_default():
            self.prepare_model(num_units)
            self.prepare_session()

    def prepare_model(self, num_units):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='input')

        with tf.name_scope('hidden'):
            w1 = tf.Variable(tf.truncated_normal([784, num_units]), name='weghts')
            b1 = tf.Variable(tf.zeros([num_units]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='hidden1')

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units, 10]), name='weghts')
            b0 = tf.Variable(tf.zeros([10]), name='biases')
            p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0, name='hidden1')

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = -tf.reduce_sum(t*tf.log(p), name='loss')
            train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        tf.scalar_summary("loss", loss)
        tf.scalar_summary("accuracy", accuracy)
        tf.histogram_summary("weights_hidden", w1)
        tf.histogram_summary("biases_hidden", b1)
        tf.histogram_summary("weights_output", w0)
        tf.histogram_summary("biases_output", b0)

        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/mnist_sl_logs", sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer

#!rm -rf /tmp/mnist_sl_logs

nn = SingleLayerNetwork(1024)
i = 0
for _ in range(2000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    nn.sess.run(nn.train_step, feed_dict={nn.x:batch_xs, nn.t:batch_ts})
    if i % 100 ==0:
        summary, loss_val, acc_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy],
                                     feed_dict={nn.x: mnist.train.images, nn.t:mnist.train.labels})
        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
        nn.writer.add_summary(summary, i)