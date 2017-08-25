import numpy as np
from modules.DataHandler import readData

phenotypes = readData("./data/16.pheno.2.txt")
print("Phenotypes:",  phenotypes.shape, "\n", phenotypes)

snps = readData("./data/16.snps.txt")
print("SNPs:",  snps.shape, "\n", snps)