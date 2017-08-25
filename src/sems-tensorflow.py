import numpy as np
from modules.DataHandler import readData

phenotypes = readData("/Users/Ryan/GitHub/SEMS-TensorFlow/src/data/pheno.2.txt")
print("Phenotypes:\n", phenotypes)
snps = readData("/Users/Ryan/GitHub/SEMS-TensorFlow/src/data/snps.txt")
print("SNPs:\n", snps)