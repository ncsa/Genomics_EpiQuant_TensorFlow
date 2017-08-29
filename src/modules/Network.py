import tensorflow as tf
import numpy as np

class ConnectedLayer:  
    def __init__(self, inSize, outSize):
        self.inSize = inSize
        self.outSize = outSize
        self.w_0 = tf.get_variable(
                                   'w_0', 
                                   shape = [self.inSize, outSize], 
                                   initializer = tf.contrib.layers.xavier_initializer()
                                  )
        self.w = tf.clip_by_value(self.w_0, 0, float("inf"))
        self.b = tf.Variable(tf.zeros(self.outSize))

    def train(self):
        self.x = tf.placeholder(tf.float32, [None, self.inSize])
        self.y = tf.placeholder(tf.float32, [None, self.outSize])
        self.z = tf.matmul(self.x, self.w) + self.b
        
        self.l2 = tf.nn.l2_loss(self.w)
        self.mse = tf.reduce_sum(tf.pow(self.y - self.z, 2)) / (2 * self.outSize)
        self.loss = self.mse + self.l2
        self.trainStep = tf.train.AdamOptimizer().minimize(self.loss)

    def shape(self):
        print("The shape of x is:", self.x.get_shape())
        print("The shape of y is:", self.y.get_shape())
        print("The shape of w is:", self.w.get_shape())
        print("The shape of b is:", self.b.get_shape())
        print()