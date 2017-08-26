import tensorflow as tf
import numpy as np

class Layer:
    def __init__(self, size):
        self.size = size
        self.w = tf.get_variable(
                                 'w', 
                                 shape = [None, self.size], 
                                 initializer = tf.contrib.layers.xavier_initializer()
                                )
        self.b = tf.Variable(tf.zeros(self.size))