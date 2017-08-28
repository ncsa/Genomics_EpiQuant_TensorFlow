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

print(
    sess.run(
        layer.squareDifference,
        feed_dict = {
            layer.x: [snpData[0]],
            layer.y: [phenoData[0]]
        }
    )
)