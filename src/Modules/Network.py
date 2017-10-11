""" Network.py

Class for neural network for multiple linear regression.
"""

import tensorflow as tf

class ConnectedLayer:
    """ A self connected layer performing multiple linear regression. """
    def __init__(self, in_size, out_size, num_batches, beta, train_rate):
        """ Initializes the tensors and sets up the graph structure.
        Performs l2 regularization, calculates root mean squared error,
        and minimizes loss using an Adam Optimizer.

        Args:
            in_size: The input size of the data.
            out_size: The ouput size of the data.

        Returns:
            None
        """
        # Initialize weight, bias, input, and observed
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.weight = tf.get_variable('weight', shape=[in_size, out_size],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=regularizer)

        self.bias = tf.Variable(tf.zeros(out_size), dtype=tf.float32)

        self.input = tf.placeholder(tf.float32, [None, in_size])
        self.observed = tf.placeholder(tf.float32, [None, out_size])

        # Calculate predicted values and loss using MSE
        predicted = tf.matmul(self.input, self.weight) + self.bias
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self.loss = tf.reduce_sum(tf.pow(self.observed - predicted, 2)) / out_size\
                    + reg_losses * beta

        # Accumulate all gradients from each batch then apply them all at once.
        opt = tf.train.GradientDescentOptimizer(train_rate)
        # Get all trainable variables and create zeros of their counterparts
        t_vars = tf.trainable_variables()
        accum_tvars = [tf.Variable(tf.zeros_like(t_var.initialized_value()), trainable=False) for t_var in t_vars]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]
        # Compute gradients for each loss and apply their contribution to the accumulator
        batch_grads_vars = opt.compute_gradients(self.loss, t_vars)
        self.accum_ops = [accum_tvars[i].assign_add(batch_grad_var[0]) for i, batch_grad_var in enumerate(batch_grads_vars)]
        # Apply the averaged gradients to update free variables.
        self.train_step = opt.apply_gradients([(accum_tvars[i] / num_batches, batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)])

    def shape(self):
        """ Prints the graph's tensor dimensions. """
        print("The shape of x is:", self.input.get_shape())
        print("The shape of y is:", self.observed.get_shape())
        print("The shape of w is:", self.weight.get_shape())
        print("The shape of b is:", self.bias.get_shape())
        print()
