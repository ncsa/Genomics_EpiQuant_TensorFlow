import tensorflow as tf

def startSession():
    """ Starts an interactive session to run the tensorflow graph. """
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    return sess