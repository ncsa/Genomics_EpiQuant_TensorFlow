import tensorflow as tf
import numpy as np

class InputLayer:
    def __init__(self, size):
        self.size = size
        self.i = tf.placeholder(tf.float32, [None, self.size])

    def shape(self):
        print("\nThe shape of the input is:", self.i.get_shape())
        print()

class ConnectedLayer:  
    def __init__(self, size):
        self.size = size
        self.w = tf.get_variable(
                                 'w', 
                                 shape = [self.size, 1], 
                                 initializer = tf.contrib.layers.xavier_initializer()
                                )
        self.b = tf.Variable(tf.zeros(self.size))

    def train(self):
        self.y = tf.placeholder(tf.float32, [None, self.size])
        self.x = tf.placeholder(tf.float32, [None, self.size])
        self.z = tf.nn.relu(tf.matmul(self.x, self.w) + self.b)
        
        self.l2 = tf.nn.l2_loss(self.w)
        self.squareDifference = tf.reduce_sum(tf.square(self.y - self.z))
        self.loss = self.squareDifference + self.l2
        self.trainStep = tf.train.AdamOptimizer().minimize(self.loss)

    def shape(self):
        print("The shape of w is:", self.w.get_shape())
        print("The shape of x is:", self.x.get_shape())
        print("The shape of b is:", self.b.get_shape())
        print()