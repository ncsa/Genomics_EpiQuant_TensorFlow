import tensorflow as tf
import numpy as np
import modules.DataHandler as dh
import modules.Network as net

print()
phenoNames, phenoData = dh.getData("./data/8.pheno.2.txt")
print("Phenotypes:", phenoData.shape, "\n\n", phenoNames, "\n\n", phenoData, "\n")

snpNames, snpData = dh.getData("./data/8.snps.txt")
print("SNPs:", snpData.shape, "\n\n", snpNames, "\n\n", snpData, "\n")

print(len(snpData[0]))
layer = net.Layer()