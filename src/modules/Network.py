import tensorflow as tf
import numpy as np

""" A self connected layer performing multiple linear regression. """
class ConnectedLayer:
    def __init__(self, inSize, outSize):
        """ Initializes the tensors and sets up the graph structure.
        Args:
            inSize: The input size of the data.
            outSize: The ouput size of the data.

        Returns:
            None
        """
        self.inSize = inSize
        self.outSize = outSize
        self.w = tf.clip_by_value(tf.Variable(tf.ones([self.inSize, self.outSize]), dtype=tf.float32), 0, float("inf"))
        self.b = tf.Variable(tf.zeros(self.outSize), dtype=tf.float32)

        self.x = tf.placeholder(tf.float32, [None, self.inSize])
        self.y = tf.placeholder(tf.float32, [None, self.outSize])
        
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
        self.rmse = tf.sqrt(tf.reduce_sum(tf.pow(self.y - self.z, 2)) / self.outSize)
        self.loss = self.rmse + self.l2
        # self.trainStep = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)
        self.trainStep = tf.train.AdamOptimizer.minimize(self.loss)


    def shape(self):
        """ Prints the graph's tensor dimensions. """
        print("The shape of x is:", self.x.get_shape())
        print("The shape of y is:", self.y.get_shape())
        print("The shape of w is:", self.w.get_shape())
        print("The shape of b is:", self.b.get_shape())
        print()