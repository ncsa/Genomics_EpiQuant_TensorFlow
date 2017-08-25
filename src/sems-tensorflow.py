import numpy as np
from modules.DataHandler import readData

phenoNames, phenoData = readData("./data/8.pheno.2.txt")
print("Phenotypes:", phenoData.shape, "\n", phenoNames, "\n", phenoData)

snpNames, snpData = readData("./data/8.snps.txt")
print("SNPs:", snpData.shape, "\n", snpNames, "\n", snpData)