import numpy as np
import codecs

def readData(filePath):
    with open(filePath) as file:
        # Reads in data from a file.
        # Strips white space and newlines from the ends.
        # Splits by newlines.
        # For each split, split by tabs.
        # Delete the first column.
        # Convert array to floating point.
        print([x.split("\t") for x in file.read().strip().split("\n")])
        # return np.delete([x.split("\t") for x in file.read().strip().split("\n")], 0, 1).astype(np.float)