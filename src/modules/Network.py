import tensorflow as tf
import numpy as np

class InputLayer:
    def __init__(self, size):
        self.size = size
        self.i = tf.placeholder(tf.float32, [None, self.size])

    def shape(self):
        print("\n", "The shape of the input is:", self.i.get_shape(), end="")

class ConnectedLayer:  
    def __init__(self, size):
        self.size = size
        self.w = tf.get_variable(
                                 'w', 
                                 shape = [self.size, 1], 
                                 initializer = tf.contrib.layers.xavier_initializer()
                                )
        self.b = tf.Variable(tf.zeros(1))

    def train(self):
        self.input = tf.placeholder(tf.float32, [None, self.size])
        self.x = tf.placeholder(tf.float32, [None, self.size])
        self.y = tf.matmul(self.x, self.w) + self.b
        self.z = tf.nn.relu(self.y)
        
        self.l2 = tf.nn.l2_loss(self.w)
        self.squareDifference = tf.reduce_sum(tf.square(self.input - self.z))
        self.loss = self.squareDifference + self.l2_loss
        self.trainStep = tf.train.AdamOptimizer().minimize(self.loss)

    def shape(self):
        print("The shape of w is:", self.w.get_shape(), end="")
        print("The shape of x is:", self.x.get_shape(), end="")
        print("The shape of b is:", self.b.get_shape(), end="")