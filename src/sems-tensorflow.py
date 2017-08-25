import numpy as np
import modules.DataHandler as dh

phenoNames, phenoData = dh.readData("./data/8.pheno.2.txt")
print("Phenotypes:", phenoData.shape, "\n", phenoNames, "\n", phenoData, "\n")

snpNames, snpData = dh.readData("./data/8.snps.txt")
print("SNPs:", snpData.shape, "\n", snpNames, "\n", snpData, "\n")

dh.calculateInteractions(phenoNames)
# dh.calculateInteractions(phenoData)

dh.calculateInteractions(snpNames)
# dh.calculateInteractions(snpData)