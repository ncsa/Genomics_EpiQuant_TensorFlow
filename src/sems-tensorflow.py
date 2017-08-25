import numpy as np
from modules.DataHandler import readData

phenotypes = readData("./data/pheno.2.txt")
print("Phenotypes:\n", phenotypes)
snps = readData("./data/snps.txt")
print("SNPs:\n", snps)