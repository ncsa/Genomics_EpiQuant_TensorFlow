import numpy as np

def readData(filePath):
    with open(filePath) as file:
        content = file.read().rstrip()
        stringRows = content.split("\n")
        output = None
        for i in range(len(stringRows)):
            tempRow = stringRows[i].split("\t")
            tempRow = np.reshape(np.delete(tempRow, 0).astype(np.float), (1, -1))
            if i == 0:
                output = tempRow
            else:
                output = np.append(output, tempRow, axis=0)
        return output