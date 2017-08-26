import numpy as np
import modules.DataHandler as dh

print()
phenoNames, phenoData = dh.readData("./data/8.pheno.2.txt")
print("Phenotypes:", phenoData.shape, "\n\n", phenoNames, "\n\n", phenoData, "\n")

snpNames, snpData = dh.readData("./data/8.snps.txt")
print("SNPs:", snpData.shape, "\n\n", snpNames, "\n\n", snpData, "\n")

# dh.calculateInteractions(phenoNames)
dh.calculateInteractions(phenoData)

# dh.calculateInteractions(snpNames)
dh.calculateInteractions(snpData)