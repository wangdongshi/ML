#import numpy as np
from random import *
from numpy import *
from itertools import islice

DIM = 28*28
NUM = 1000

class optStruct:
	def __init__(self,dataMatIn,classLabels,C,toler, kTup):
		self.X=dataMatIn
		self.labelMat=classLabels
		self.C=C
		self.tol=toler
		self.m=shape(dataMatIn)[0]
		self.alphas=mat(zeros((self.m,1)))
		self.b=0
		self.Cache=mat(zeros((self.m,2)))
		self.K = mat(zeros((self.m,self.m)))
		for i in range(self.m):
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

#格式化计算误差的函数，方便多次调用
def calcEk(oS, k):
	'''计算预测误差'''
	fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
	Ek = fXk - float(oS.labelMat[k])
	return Ek

#修改选择第二个变量alphaj的方法
def selectJ(i,oS,Ei):
	maxK=-1;maxDeltaE= 0 ;Ej=0
	#将误差矩阵每一行第一列置1，以此确定出误差不为0
	#的样本
	oS.Cache[i]=[1,Ei]
	#获取缓存中Ei不为0的样本对应的alpha列表
	validEcacheList=nonzero(oS.Cache[:,0].A)[0]
	#在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
	if(len(validEcacheList)>0):
		for k in validEcacheList:
			if k ==i:continue
			Ek=calcEk(oS,k)
			deltaE=abs(Ei-Ek)
			if(deltaE>maxDeltaE):
				maxK=k;maxDeltaE=deltaE;Ej=Ek
		return maxK,Ej
	else:
	#否则，就从样本集中随机选取alphaj
		j=selectJrand(i,oS.m)
		Ej=calcEk(oS,j)
	return j,Ej

#更新误差矩阵
def updateEk(oS,k):
	Ek=calcEk(oS,k)
	oS.Cache[k]=[1,Ek]

def loadDataSet(fName, startLine, endline):
	dataMat = []
	labelMat = []
	fr = open(fName)
	for line in islice(fr, startLine, endline):
		lineArr = line.strip().split(',')
		labelMat.append(int(lineArr[0]))
		dataLine = []
		for i in range(1, DIM+1):
			dataLine.append(float(lineArr[i]))
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

def kernelTrans( X, A, kTup):
	'''
	RBF kernel function
	'''
	m ,n = shape(X)
	K = mat(zeros((m,1)))
	if kTup[0] == 'lin': K = X * A.T
	elif kTup[0] == 'rbf':
		for j in range(m):
			deltaRow = X[j,:] - A
			K[j] = deltaRow * deltaRow.T
		K = exp(K / (-1 * kTup[1] ** 2))
	else: raise NameError('huston ---')
	return K

def innerL(i, oS):
	Ei = calcEk(oS, i)
	if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
		j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L==H: print("L==H"); return 0
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
		if eta >= 0: print("eta>=0"); return 0
		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
		oS.alphas[j] = updateAlpha(oS.alphas[j],H,L)
		updateEk(oS, j) #added this for the Ecache
		if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
		updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2)/2.0
		return 1
	else: return 0

def PlattSMO(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)): #full Platt SMO
	oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
	itera = 0
	entireSet = True; alphaPairsChanged = 0
	while (itera < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		if entireSet:   #go over all
			for i in range(oS.m):        
				alphaPairsChanged += innerL(i,oS)
				print("fullSet, iter: %d i:%d, pairs changed %d" % (itera,i,alphaPairsChanged))
			itera += 1
		else:#go over non-bound (railed) alphas
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print("non-bound, iter: %d i:%d, pairs changed %d" % (itera,i,alphaPairsChanged))
			itera += 1
		if entireSet: entireSet = False #toggle entire set loop
		elif (alphaPairsChanged == 0): entireSet = True  
		print("iteration number: %d" % itera)
	return oS.b,oS.alphas

def testRbf(k1=1.3):
	dataArr,labelArr = loadDataSet('16.MNIST.train.csv', 1, NUM+1)
	for i in range(0, NUM) :
		if labelArr[i] == 0 : labelArr[i] = 1
		else : labelArr[i] = -1
	b,alphas = PlattSMO(dataArr, labelArr, 1, 0.0001, 100, ('rbf', k1)) #C=200 important
	datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
	svInd=nonzero(alphas.A>0)[0]
	sVs=datMat[svInd] #get matrix of only support vectors
	labelSV = labelMat[svInd];
	print("there are %d Support Vectors" % shape(sVs)[0])
	m,n = shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict)!=sign(labelArr[i]): errorCount += 1
	print("the training error rate is: %f" % (float(errorCount)/m))
	return

	# test the last 5000 data
	dataArr,labelArr = loadDataSet('16.MNIST.train.csv', 35001, 40001)
	for i in range(0, 5000) :
		if labelArr[i] == 0 : labelArr[i] = 1
		else : labelArr[i] = -1
	errorCount = 0
	datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
	m,n = shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict)!=sign(labelArr[i]): errorCount += 1    
	print("the test error rate is: %f" % (float(errorCount)/m)) 

if __name__ == '__main__':
	testRbf()

