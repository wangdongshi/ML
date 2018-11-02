import numpy as np
from itertools import islice

def loadDataSet(fName, lineNum):
	dataMat = []
	labelMat = []
	fr = open(fName)
	for line in islice(fr, 1, lineNum):
		lineArr = line.strip().split(',')
		labelMat.append(int(lineArr[0]))
		dataLine = []
		for i in range(1, 784):
			dataLine.append(int(lineArr[i]))
		dataMat.append(dataLine)
	return dataMat, labelMat

if __name__ == '__main__':
	
	dataSet, labelSet = loadDataSet('16.MNIST.train.csv', 30000)
	print (labelSet[0], dataSet[0])
