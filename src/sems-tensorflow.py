import numpy as np
from modules.DataHandler import readData

phenotypes = readData("./data/pheno.2.txt")
print("Phenotypes:",  phenotypes.shape, "\n", phenotypes)

snps = readData("./data/snps.txt")
print("SNPs:",  snps.shape, "\n", snps)