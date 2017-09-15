import tensorflow as tf
import numpy as np
import modules.DataHandler as dh
import modules.Network as net
import modules.SessionHandler as sh
import modules.Timer as timer
import modules.BatchBuilder as bb
import modules.Progress as prog

import sys

appTime = timer.Timer()

# Gets the phenotype names and the regression y values.
print()
phenoNames, phenoData = dh.getData("./data/8.pheno.2.txt", False)
print("Phenotypes:", phenoData.shape, "\n\n", phenoNames, "\n\n", phenoData, "\n")

# Gets the snp names and the regressors.
snpNames, snpData = dh.getData("./data/8.snps.txt", True)
snpNames = np.transpose(snpNames)
snpData = np.transpose(snpData)
print("SNPs:", snpData.shape, "\n\n", snpNames, "\n\n", snpData, "\n")

# Get input and output size for tensors.
inSize = len(snpData[0])
batches = len(phenoData[0])
outSize = 1

# Make Batches out of snpData and unallocate snpData
snpDataBatches = bb.makeBatches(snpData, batches)
snpData = None

# Initialize graph structure.
layer = net.ConnectedLayer(inSize, outSize)
layer.train()
layer.shape()

# Start TensorFlow session.
sess = sh.startSession()

alpha = 0.05
pastLoss = 0
step = 1
while True:
    # Train for an epoch
    # Get the current loss and train the graph.
    for i in range(len(snpDataBatches)):
        currentLoss, _ = sess.run(
            [layer.loss, layer.trainStep],
            feed_dict = {
                layer.x: snpDataBatches[i],
                layer.y: np.asarray([phenoData[0][i]]).reshape(1, outSize)
            }
        )
        prog.progress(i, len(snpDataBatches), "Training Completed in Epoch " + str(step))

    # sh.printTensors(sess, layer, snpData, phenoData, 0)

    prog.logTraining(pastLoss, currentLoss, alpha, step, appTime)

    # Save the weight and bias tensors when the model converges.
    if abs(pastLoss - currentLoss) < (alpha):
        np.savetxt(
            "w.out",
            sess.run(
                layer.w,
                feed_dict = {
                    layer.x: snpDataBatches[0],
                    layer.y: np.asarray([phenoData[0][0]]).reshape(1, outSize)
                }
            ),
            delimiter="\t",
            fmt="%1.2e"
        )
        np.savetxt(
            "b.out",
            sess.run(
                layer.b,
                feed_dict = {
                    layer.x: snpDataBatches[0],
                    layer.y: np.asarray([phenoData[0][0]]).reshape(1, outSize)
                }
            ),
            delimiter="\t",
            fmt="%1.2e"
        )
        break
    pastLoss = currentLoss
    step += 1
    # np.random.shuffle(snpDataBatches)

print(" [", appTime.getTime(), "]", "Closing session...\n")
sess.close()