""" SemsTensorFlow.py

Builds, trains and runs a neural network.
"""

# import sys
import numpy as np
import Modules.DataHandler as dh
import Modules.Network as net
import Modules.SessionHandler as sh
import Modules.Timer as timer
import Modules.BatchBuilder as bb
import Modules.Progress as prog

OUT_SIZE = 1
ALPHA = 0.5
BETA = 0.01
TRAIN_RATE = 0.00001
KEEP_PROB = 0.1

def main():
    """ Builds, trains, and runs the neural network. """
    app_time = timer.Timer()

    # Gets the phenotype names and the regression y values.
    print()
    pheno_names, pheno_data = dh.get_data("./data/8.pheno.2.txt", False)
    print("Phenotypes:", pheno_data.shape, "\n\n", pheno_names, "\n\n", pheno_data, "\n")

    # Gets the snp names and the regressors.
    snp_names, snp_data = dh.get_data("./data/8.snps.txt", True)
    snp_names = np.transpose(snp_names)
    snp_data = np.transpose(snp_data)
    print("SNPs:", snp_data.shape, "\n\n", snp_names, "\n\n", snp_data, "\n")

    # Make batches out of snp_data and unallocate snp_data
    snp_data = bb.make_batches(snp_data, len(pheno_data[0]))

    # Initialize graph structure.
    layer = net.ConnectedLayer(len(snp_data[0][0]), OUT_SIZE, len(snp_data), BETA, TRAIN_RATE)
    layer.shape()

    # Start TensorFlow session.
    sess = sh.start_session()

    past_accuracy = 0
    past_loss = 0
    step = 1
    index = np.arange(len(snp_data))

    while True:
        # Zero gradient accumulators
        sess.run(layer.zero_ops)

        # Generate randomized index
        np.random.shuffle(index)
        snp_sample = None
        pheno_sample = None
        count = 0

        # Accumulate gradients
        for i in range(len(snp_data)):
            snp_sample = snp_data[index[i]]
            pheno_sample = [[pheno_data[0][index[i]]]]
            current_loss, _ = sess.run(
                [layer.loss, layer.accum_ops],
                feed_dict={
                    layer.input: snp_sample,
                    layer.observed: pheno_sample,
                    layer.keep_prob: [[KEEP_PROB]]
                }
            )

            if current_loss < ALPHA:
                count += 1

            # prog.progress(i, len(snp_data), "Training Completed in Epoch " + str(step))

        # Apply averaged gradient and calculate current loss
        current_loss, _ = sess.run(
            [layer.loss, layer.train_step],
            feed_dict={
                layer.input: snp_sample,
                layer.observed: pheno_sample,
                layer.keep_prob: [[KEEP_PROB]]
            }
        )

        accuracy = count / len(snp_data)
        prog.log_training(accuracy, past_accuracy, current_loss, past_loss, ALPHA, step, app_time)

        # Save the weight and bias tensors when the model converges.
        if abs(past_accuracy - accuracy) < 0.0005:
            np.savetxt(
                "w.out",
                sess.run(
                    layer.weight,
                    feed_dict={
                        layer.input: snp_data[0],
                        layer.observed: np.asarray([pheno_data[0][0]]).reshape(1, OUT_SIZE),
                        layer.keep_prob: [[1]]
                    }
                ),
                delimiter="\t"
            )
            np.savetxt(
                "b.out",
                sess.run(
                    layer.bias,
                    feed_dict={
                        layer.input: snp_data[0],
                        layer.observed: np.asarray([pheno_data[0][0]]).reshape(1, OUT_SIZE),
                        layer.keep_prob: [[1]]
                    }
                ),
                delimiter="\t"
            )
            break
        past_accuracy = accuracy
        past_loss = current_loss
        step += 1

    print(" [", app_time.get_time(), "]", "Closing session...\n")
    sess.close()

if __name__ == "__main__":
    main()
