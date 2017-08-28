import tensorflow as tf
import numpy as np
import modules.DataHandler as dh
import modules.Network as net
import modules.SessionHandler as sh

print()
phenoNames, phenoData = dh.getData("./data/8.pheno.2.txt", False)
print("Phenotypes:", phenoData.shape, "\n\n", phenoNames, "\n\n", phenoData, "\n")

snpNames, snpData = dh.getData("./data/8.snps.txt", True)
snpNames = np.transpose(snpNames)
snpData = np.transpose(snpData)
print("SNPs:", snpData.shape, "\n\n", snpNames, "\n\n", snpData, "\n")

inSize = len(snpData[0])
outSize = len(phenoData[0])

layer = net.ConnectedLayer(inSize, outSize)
layer.train()
layer.shape()

sess = sh.startSession()

pastLoss = 0
while True:
    currentLoss = sess.run(
        layer.loss,
        feed_dict = {
            layer.x: [snpData[0]],
            layer.y: [phenoData[0]]
        }
    )
    print(
        sess.run(
            layer.w,
            feed_dict = {
                layer.x: [snpData[0]],
                layer.y: [phenoData[0]]
            }
        ), 
        "\n",
        sess.run(
            layer.b,
            feed_dict = {
                layer.x: [snpData[0]],
                layer.y: [phenoData[0]]
            }
        )
    )
    sess.run(
        layer.trainStep,
        feed_dict = {
            layer.x: [snpData[0]],
            layer.y: [phenoData[0]]
        }
    )
    print("Current loss:", "{:10.2f}".format(currentLoss))
    print("Loss Difference:", "{:10.2f}".format(pastLoss-currentLoss))
    if Math.abs(pastLoss - currentLoss) < 0.05:
        break
    pastLoss = currentLoss
print("Closing session...")