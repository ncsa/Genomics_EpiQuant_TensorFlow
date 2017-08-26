import tensorflow as tf
import numpy as np

class InputLayer:
    def __init__(self, size):
        self.size = size
        self.i = tf.placeholder(tf.float32, [None, self.size])

    def shape(self):
        print("\n", "The shape of the input is:", self.i.get_shape())

class ConnectedLayer:
    def __init__(self, size):
        self.size = size
        self.w = tf.get_variable(
                                 'w', 
                                 shape = [self.size, self.size], 
                                 initializer = tf.contrib.layers.xavier_initializer()
                                )
        self.b = tf.Variable(tf.zeros(self.size))

    def shape(self):
        print("\n", "The shape of the w is:", self.w.get_shape())
        print("\n", "The shape of the b is:", self.b.get_shape())