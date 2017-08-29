import tensorflow as tf
import numpy as np
import modules.DataHandler as dh
import modules.Network as net
import modules.SessionHandler as sh

# Gets the phenotype names and the regression y values.
print()
phenoNames, phenoData = dh.getData("./data/8.pheno.2.txt", False)
print("Phenotypes:", phenoData.shape, "\n\n", phenoNames, "\n\n", phenoData, "\n")

# Gets the snp names and the regressors.
snpNames, snpData = dh.getData("./data/8.snps.txt", True)
snpNames = np.transpose(snpNames)
snpData = np.transpose(snpData)
print("SNPs:", snpData.shape, "\n\n", snpNames, "\n\n", snpData, "\n")

inSize = len(snpData[0])
outSize = len(phenoData[0])

# Initialize graph structure.
layer = net.ConnectedLayer(inSize, outSize)
layer.train()
layer.shape()

# Start TensorFlow session.
sess = sh.startSession()

pastLoss = 0
while True:
    # Get the current loss of the graph.
    currentLoss = sess.run(
        layer.loss,
        feed_dict = {
            layer.x: snpData,
            layer.y: [phenoData[0]]
        }
    )

    sh.printTensors(sess, layer, 0)

    print()
    print("  Loss:", "{:10.2f}".format(currentLoss))
    print(" Delta:", "{:10.2f}".format(abs(pastLoss-currentLoss)))
    print(" Alpha:", "{:10.2f}".format(currentLoss * 0.0001))

    # Save the weight and bias tensors when the model converges.
    if abs(pastLoss - currentLoss) < (currentLoss * 0.0001):
        np.savetxt(
            "w.out",
            sess.run(
                layer.w,
                feed_dict = {
                    layer.x: snpData,
                    layer.y: [phenoData[0]]
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
                    layer.x: snpData,
                    layer.y: [phenoData[0]]
                }
            ),
            delimiter="\t",
            fmt="%1.2e"
        )
        break
    pastLoss = currentLoss

    # Train the model.
    sess.run(
        layer.trainStep,
        feed_dict = {
            layer.x: snpData,
            layer.y: [phenoData[0]]
        }
    )

print("Closing session...\n")
sess.close()