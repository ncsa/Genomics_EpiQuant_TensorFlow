""" SessionHandler.py

Starts sessions and prints tensors.
"""
import tensorflow as tf

def start_session():
    """ Starts an interactive session to run the tensorflow graph. """
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    # tf.global_variables_initializer().run()
    sess.run(tf.initialize_all_variables())
    return sess

def print_tensors(sess, layer, snp_data, pheno_data, i):
    """ Prints intermediate tensors. """
    w, b = sess.run(
        [layer.w, layer.b],
        feed_dict={
            layer.x: snp_data,
            layer.y: [pheno_data[i]]
        }
    )
    print(w)
    print(b)
