#import numpy as np
from random import *
from numpy import *
from itertools import islice

DIM = 28*28
NUM = 1000

def loadDataSet(fName, lineNum):
	dataMat = []
	labelMat = []
	fr = open(fName)
	for line in islice(fr, 1, lineNum+1):
		lineArr = line.strip().split(',')
		labelMat.append(int(lineArr[0]))
		dataLine = []
		for i in range(1, DIM+1):
			dataLine.append(int(lineArr[i]))
		dataMat.append(dataLine)
	return dataMat, labelMat

def selectRand(i, m):
	j = i
	while(j==i):
		j = int(random.uniform(0,m))
	return j

def updateAlpha(alphaJ, upper, lower):
	if alphaJ > upper: alphaJ = upper
	if alphaJ < lower: alphaJ = lower
	return alphaJ

def SimpleSMO(dataSet, labelSet, C, toler, maxIter):
	matrix = mat(dataSet)
	label = mat(labelSet).transpose()
	#print (labelMatrix, dataMatrix)
	b = 0
	m,n = shape(matrix)
	print ("m=%d; n=%d" %(m,n))
	alphaM = mat(zeros((m,1)))
	itera = 0
	while (itera < maxIter):
		alphaChanged = 0
		for i in range(m):
			fXi = float(multiply(alphaM, label).T*(matrix*matrix[i,:].T))+b
			Ei = fXi-float(label[i])
			if ((label[i]*Ei<-toler) and (alphaM[i]<C)) or ((label[i]*Ei>toler) and alphaM[i]>0):
				j = selectRand(i, m)
				fXj = float(multiply(alphaM, label).T*(matrix*matrix[j,:].T))+b
				Ej = fXj-float(label[j])
				alphaIold = alphaM[i].copy()
				alphaJold = alphaM[j].copy()
				if label[i] != label[j]:
					L = max(0, alphaM[j]-alphaM[i])
					H = max(C, C+alphaM[j]-alphaM[i])
				else:
					L = max(0, alphaM[j]+alphaM[i]-C)
					H = max(C, alphaM[j]+alphaM[i])
				if L==H: print("L==H"); continue
				eta = 2.0*matrix[i,:]*matrix[j,:].T - matrix[i,:]*matrix[i,:].T - matrix[j,:]*matrix[j,:].T
				if eta >= 0:print("eta>=0"); continue
				alphaM[j] -= label[j] * (Ei-Ej) / eta
				alphaM[j] = updateAlpha(alphaM[j], H, L)
				if (abs(alphaM[j] - alphaJold) < 0.00001):
					print("j not moving enough")
					continue
				alphaM[i] += label[j]*label[i]*(alphaJold-alphaM[j])
				b1 = b - Ei - label[i]*(alphaM[i]-alphaIold)*matrix[i,:]*matrix[i,:].T - label[j]*(alphaM[j]-alphaJold)*matrix[i,:]*matrix[j,:].T
				b2 = b - Ej - label[i]*(alphaM[i]-alphaIold)*matrix[i,:]*matrix[j,:].T - label[j]*(alphaM[j]-alphaJold)*matrix[j,:]*matrix[j,:].T
				if (0<alphaM[i]) and (C>alphaM[i]): b = b1
				elif (0<alphaM[j]) and (C>alphaM[j]): b = b2
				else: b = (b1+b2)/2.0
				alphaChanged += 1
				print("iter: %d i:%d, pairs changed %d" %(itera,i,alphaChanged))
		if alphaChanged == 0 : itera += 1
		else : itera = 0
		print("iteration number : %d" %itera)
	return b, alphaM

if __name__ == '__main__':
	dataSet, labelSet = loadDataSet('16.MNIST.train.csv', NUM)
	#print(labelSet[0], dataSet)
	for i in range(0, NUM) :
		if labelSet[i] == 0 : labelSet[i] = 1
		else : labelSet[i] = -1
	#print(labelSet, dataSet[100])

	b, alpha = SimpleSMO(dataSet, labelSet, 0.6, 0.001, 40)
	print("b=%d" %b)
	print(alpha)

