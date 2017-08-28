import tensorflow as tf
import numpy as np

class ConnectedLayer:  
    def __init__(self, inSize, outSize):
        self.inSize = inSize
        self.outSize = outSize
        self.w = tf.get_variable(
                                 'w', 
                                 shape = [self.inSize, outSize], 
                                 initializer = tf.contrib.layers.xavier_initializer()
                                )
        self.b = tf.Variable(tf.zeros(self.outSize))

    def train(self):
        self.x = tf.placeholder(tf.float32, [None, self.inSize])
        self.y = tf.placeholder(tf.float32, [None, self.outSize])
        self.z = tf.nn.relu(tf.matmul(self.x, self.w) + self.b)
        
        self.l2 = tf.nn.l2_loss(self.w)
        self.squareDifference = tf.reduce_sum(tf.square(self.y - self.z))
        self.loss = self.squareDifference + self.l2
        self.trainStep = tf.train.AdamOptimizer().minimize(self.loss)

    def shape(self):
        print("The shape of x is:", self.x.get_shape())
        print("The shape of y is:", self.y.get_shape())
        print("The shape of w is:", self.w.get_shape())
        print("The shape of b is:", self.b.get_shape())
        print()