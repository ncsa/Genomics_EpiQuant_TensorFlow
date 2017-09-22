""" Builds, trains and runs a neural network. """

import numpy as np
import Modules.DataHandler as dh
import Modules.Network as net
import Modules.SessionHandler as sh
import Modules.Timer as timer
import Modules.BatchBuilder as bb
import Modules.Progress as prog

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

    # Get input and output size for tensors.
    in_size = len(snp_data[0])
    batches = len(pheno_data[0])
    out_size = 1

    # Make Batches out of snp_data and unallocate snp_data
    snp_data = bb.make_batches(snp_data, batches)

    # Initialize graph structure.
    layer = net.ConnectedLayer(in_size, out_size)
    layer.train()
    layer.shape()

    # Start TensorFlow session.
    sess = sh.start_session()

    alpha = 0.05
    past_loss = 0
    step = 1
    while True:
        # Train for an epoch
        # Get the current loss and train the graph.
        for i in range(len(snp_data)):
            current_loss, _ = sess.run(
                [layer.loss, layer.train_step],
                feed_dict={
                    layer.x: snp_data[i],
                    layer.y: np.asarray([pheno_data[0][i]]).reshape(1, out_size)
                }
            )
            prog.progress(i, len(snp_data), "Training Completed in Epoch " + str(step))

        # sh.print_tensors(sess, layer, snp_data, pheno_data, 0)

        prog.log_training(past_loss, current_loss, alpha, step, app_time)

        # Save the weight and bias tensors when the model converges.
        if abs(past_loss - current_loss) < (alpha):
            np.savetxt(
                "w.out",
                sess.run(
                    layer.w,
                    feed_dict={
                        layer.x: snp_data[0],
                        layer.y: np.asarray([pheno_data[0][0]]).reshape(1, out_size)
                    }
                ),
                delimiter="\t",
                fmt="%1.2e"
            )
            np.savetxt(
                "b.out",
                sess.run(
                    layer.b,
                    feed_dict={
                        layer.x: snp_data[0],
                        layer.y: np.asarray([pheno_data[0][0]]).reshape(1, out_size)
                    }
                ),
                delimiter="\t",
                fmt="%1.2e"
            )
            break
        past_loss = current_loss
        step += 1

        rng_state = np.random.get_state()
        np.random.shuffle(snp_data)
        np.random.set_state(rng_state)
        np.random.shuffle(pheno_data[0])

    print(" [", app_time.get_time(), "]", "Closing session...\n")
    sess.close()

if __name__ == "__main__":
    main()
