""" Network.py

Class for neural network for multiple linear regression.
"""

import tensorflow as tf

class ConnectedLayer:
    """ A self connected layer performing multiple linear regression. """
    def __init__(self, in_size, out_size):
        """ Initializes the tensors and sets up the graph structure.
        Args:
            in_size: The input size of the data.
            out_size: The ouput size of the data.

        Returns:
            None
        """
        self.in_size = in_size
        self.out_size = out_size
        self.w = tf.clip_by_value(tf.Variable(tf.ones([self.in_size,
                                                       self.out_size]),
                                              dtype=tf.float32),
                                  0,
                                  float("inf"))
        self.b = tf.Variable(tf.zeros(self.out_size), dtype=tf.float32)

        self.x = tf.placeholder(tf.float32, [None, self.in_size])
        self.y = tf.placeholder(tf.float32, [None, self.out_size])

        self.z = tf.matmul(self.x, self.w) + self.b

    def train(self):
        """ Performs l2 regularization, calculates root mean squared error, and minimizes loss
        using an Adam Optimizer.

        Args:
            None

        Returns:
            None
        """
        self.l2 = tf.nn.l2_loss(self.w)
        self.rmse = tf.sqrt(tf.reduce_sum(tf.pow(self.y - self.z, 2)) / self.out_size)
        self.loss = self.rmse + self.l2
        self.train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(self.loss)

    def shape(self):
        """ Prints the graph's tensor dimensions. """
        print("The shape of x is:", self.x.get_shape())
        print("The shape of y is:", self.y.get_shape())
        print("The shape of w is:", self.w.get_shape())
        print("The shape of b is:", self.b.get_shape())
        print()
