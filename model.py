import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Model:

    def __init__(self):
        self.epsilon = 0.01

    def initialize(self, outfile):
        tf.reset_default_graph()

        n_inputs = 301
        n_outputs = 1

        weights = tf.Variable(tf.truncated_normal([n_inputs, n_outputs]))
        biases = tf.Variable(tf.zeros([n_outputs]))

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            saver.save(sess, outfile)

    def transfer(self, infile, outfile):
        n_inputs = 301
        n_outputs = 1

        weights = tf.Variable(tf.truncated_normal([n_inputs, n_outputs]))
        biases = tf.Variable(tf.zeros([n_outputs]))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, infile)

            saver.save(sess, outfile)

    def train(self, infile, outfile, inputs, outputs):

        tf.reset_default_graph()

        learning_rate = 0.01
        n_inputs = 301
        n_outputs = 1

        features = tf.placeholder(tf.float32, [None, n_inputs])
        labels = tf.placeholder(tf.float32, [None, n_outputs])

        weights = tf.Variable(tf.truncated_normal([n_inputs, n_outputs]))
        biases = tf.Variable(tf.zeros([n_outputs]))

        logits = tf.add(tf.matmul(features, weights), biases)

        cost = tf.reduce_sum(tf.pow(logits-labels, 2))/(2*1)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, infile)

            sess.run(optimizer, feed_dict={features: inputs, labels: outputs})

            saver.save(sess, outfile)

    def fit(self, infile, inputs):
        tf.reset_default_graph()

        n_inputs = 301
        n_outputs = 1

        features = tf.placeholder(tf.float32, [None, n_inputs])
        labels = tf.placeholder(tf.float32, [None, n_outputs])

        weights = tf.Variable(tf.truncated_normal([n_inputs, n_outputs]))
        biases = tf.Variable(tf.zeros([n_outputs]))

        logits = tf.add(tf.matmul(features, weights), biases)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, infile)

            prediction = sess.run(logits, feed_dict={features: inputs})
            return prediction


# layout - check
# communication
# saving - check
# data
# model
