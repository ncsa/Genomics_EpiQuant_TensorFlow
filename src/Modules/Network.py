""" Network.py

Class for neural network for multiple linear regression.
"""

import tensorflow as tf

class ConnectedLayer:
    """ A self connected layer performing multiple linear regression. """
    def __init__(self, in_size, out_size):
        """ Initializes the tensors and sets up the graph structure.
        Performs l2 regularization, calculates root mean squared error,
        and minimizes loss using an Adam Optimizer.

        Args:
            in_size: The input size of the data.
            out_size: The ouput size of the data.

        Returns:
            None
        """
        self.weight = tf.get_variable('weight', shape=[in_size, out_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
        self.bias = tf.Variable(tf.zeros(out_size), dtype=tf.float32)

        self.input = tf.placeholder(tf.float32, [None, in_size])
        self.observed = tf.placeholder(tf.float32, [None, out_size])

        self.predicted = tf.matmul(self.input, self.weight) + self.bias

        self.loss = tf.sqrt(tf.reduce_sum(tf.pow(self.observed - self.predicted, 2)) / out_size)\
                    + tf.nn.l2_loss(self.weight) * 100
        self.train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(self.loss)

    def shape(self):
        """ Prints the graph's tensor dimensions. """
        print("The shape of x is:", self.input.get_shape())
        print("The shape of y is:", self.observed.get_shape())
        print("The shape of w is:", self.weight.get_shape())
        print("The shape of b is:", self.bias.get_shape())
        print()
