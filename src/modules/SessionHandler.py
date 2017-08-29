import tensorflow as tf

def startSession():
    """ Starts an interactive session to run the tensorflow graph. """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config = config)
    tf.global_variables_initializer().run()
    return sess

def printTensors(sess, i):
    """ Prints intermediate tensors. """
    print(
        sess.run(
            layer.w,
            feed_dict = {
                layer.x: snpData,
                layer.y: [phenoData[i]]
            }
        )
    )
    print(
        sess.run(
            layer.b,
            feed_dict = {
                layer.x: snpData,
                layer.y: [phenoData[i]]
            }
        )
    )