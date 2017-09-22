""" SemsTensorFlow.py

Builds, trains and runs a neural network.
"""

import sys
import numpy as np
import Modules.DataHandler as dh
import Modules.Network as net
import Modules.SessionHandler as sh
import Modules.Timer as timer
import Modules.BatchBuilder as bb
import Modules.Progress as prog

OUT_SIZE = 1
ALPHA = 0.05

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

    # Make Batches out of snp_data and unallocate snp_data
    snp_data = bb.make_batches(snp_data, len(pheno_data[0]))

    # Initialize graph structure.
    layer = net.ConnectedLayer(len(snp_data[0]), OUT_SIZE)
    layer.shape()

    # Start TensorFlow session.
    sess = sh.start_session()

    past_loss = 0
    step = 1

    while True:
        print(len(snp_data))
        print(len(pheno_data[0]))
        sys.exit()
        # rng_state = np.random.get_state()
        # np.random.shuffle(snp_data)
        # np.random.set_state(rng_state)
        # np.random.shuffle(pheno_data[0])

        # Train for an epoch
        # Get the current loss and train the graph.
        for i in range(len(snp_data)):
            current_loss, _ = sess.run(
                [layer.loss, layer.train_step],
                feed_dict={
                    layer.input: snp_data[i],
                    layer.observed: np.asarray([pheno_data[0][i]]).reshape(1, OUT_SIZE)
                }
            )
            prog.progress(i, len(snp_data), "Training Completed in Epoch " + str(step))

        prog.log_training(past_loss, current_loss, ALPHA, step, app_time)

        # Save the weight and bias tensors when the model converges.
        if abs(past_loss - current_loss) < (ALPHA):
            np.savetxt(
                "w.out",
                sess.run(
                    layer.weight,
                    feed_dict={
                        layer.input: snp_data[0],
                        layer.observed: np.asarray([pheno_data[0][0]]).reshape(1, OUT_SIZE)
                    }
                ),
                delimiter="\t",
                fmt="%1.2e"
            )
            np.savetxt(
                "b.out",
                sess.run(
                    layer.bias,
                    feed_dict={
                        layer.input: snp_data[0],
                        layer.observed: np.asarray([pheno_data[0][0]]).reshape(1, OUT_SIZE)
                    }
                ),
                delimiter="\t",
                fmt="%1.2e"
            )
            break
        past_loss = current_loss
        step += 1

    print(" [", app_time.get_time(), "]", "Closing session...\n")
    sess.close()

if __name__ == "__main__":
    main()
