import tensorflow as tf
import os

def startSession():
    """ Starts an interactive session to run the tensorflow graph. """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config = config)
    tf.global_variables_initializer().run()
    return sess

def progress(i, length, message):
    print(" [", "{:6.2f}".format((i + 1) / length * 100) + "%", "]", message, end="\r")
    if i + 1 == length:
        print("\n")

def logTraining(pastLoss, currentLoss, alpha, step, appTime):
    print(
        "[", appTime.getTime(), "]",
        "   Step:", "{:8d}".format(step),
        "   Loss:", "{:.2E}".format(currentLoss),
        "   Delta:", "{:.2E}".format(abs(pastLoss-currentLoss)),
        "   Alpha:", "{:.2E}".format(alpha)
    )

def printTensors(sess, layer, snpData, phenoData, i):
    """ Prints intermediate tensors. """
    w, b = sess.run(
        [layer.w, layer.b],
        feed_dict = {
            layer.x: snpData,
            layer.y: [phenoData[i]]
        }
    )
    print(w)
    print(b)