#!/usr/bin/python
#-*-coding:utf-8
from numpy import *
import pandas as pd
import os
import csv
import pdb

def arrangeCsv(fileIn, fileOut):
	#dimens = ['101', '109_14', '110_14', '127_14', '150_14', \
	#		'121', '122', '124', '125', '126', '127', '128', '129', \
	#		'205', '206', '207', '210', '216', \
	#		'508', '509', '702', '853', '301']
	dimens = ['508', '509', '702', '853']
	matrix = []
	labels = []
	# set index info
	dimIndex = {}
	for i,s in enumerate(dimens):
		dimIndex[s] = i
	#pdb.set_trace()
	# dump csv file info
	csvFile = open(fileIn, "r")
	reader = csv.reader(csvFile)
	if os.path.exists(fileOut): os.remove(fileOut)
	outFile = open(fileOut,'a')
	cnt = 0
	for i,rows in enumerate(reader):
		#if i > 10000: break
		record = [1.0] * len(dimens)
		char = rows[5].split('\x01')
		for s in char:
			pair = s.split('\x02')
			if pair[0] in dimens:
				record[dimIndex[pair[0]]] = float(s.split('\x03')[1])
		all_one = True
		for j in record:
			#pdb.set_trace()
			if j != 1.0: all_one = False; break
		if (not all_one) and (rows[1] == '1'):
			cnt += 1
			matrix.append(record)
			#pdb.set_trace()
			if (rows[1] == '1' and rows[2] == '1'): labels.append(+1)
			else: labels.append(-1)
		if i != 0 and i%10000 == 0:
			# write csv file info into output file
			test = pd.DataFrame(columns = dimens, index = labels, data = matrix)
			test.to_csv(outFile, encoding = 'utf8', header = False)
			matrix = []
			labels = []
			print('input record number : %d, output record number : %d' %(i, cnt), end='\r')
	outFile.close()
	csvFile.close()
	
def loadSimpData():
	datMat = matrix([[ 1. ,  2.1],
		[ 2. ,  1.1],
		[ 1.3,  1. ],
		[ 1. ,  1. ],
		[ 2. ,  1. ]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return datMat,classLabels

def loadDataSet(fileName):	  #general function to parse tab -delimited floats
	numFeat = len(open(fileName).readline().split('\t')) #get number of fields
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	fr.close()
	return dataMat,labelMat

def loadSkeletonData(fileName):
	dimens = ['508', '509', '702', '853']
	dataMat = []; labelMat = []
	csvFile = open(fileName, "r")
	reader = csv.reader(csvFile)
	minus = 0; plus = 0
	for i,rows in enumerate(reader):
		fRows = [float(x) for x in rows[1:5]]
		#pdb.set_trace()
		if rows[0] == '-1': minus += 1
		else: plus += 1
		if minus > 5500: minus -= 1; continue
		dataMat.append(fRows)
		labelMat.append(float(rows[0]))
		if (minus+plus)%1000 == 0: print('data number : %d' %(minus+plus), end='\r')
		#if i%10000 == 0: print('data number : %d' %(i), end='\r')
	csvFile.close()
	#pdb.set_trace()
	outFile = open('temp.csv', "w")
	test = pd.DataFrame(columns = dimens, index = labelMat, data = dataMat)
	test.to_csv(outFile, encoding = 'utf8', header = False)
	return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray

def buildStump(dataArr,classLabels,D):
	dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
	m,n = shape(dataMatrix)
	numSteps = 100.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
	minError = inf #init error sum, to +infinity
	for i in range(n):#loop over all dimensions
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
		stepSize = (rangeMax-rangeMin)/numSteps
		for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
			for inequal in ['lt', 'gt']: #go over less than and greater than
				threshVal = (rangeMin + float(j) * stepSize)
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0
				weightedError = D.T*errArr  #calc total error multiplied by D
				#print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" %(i, threshVal, inequal, weightedError))
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
	weakClassArr = []
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/m)   #init D to all equal
	aggClassEst = mat(zeros((m,1)))
	for i in range(numIt):
		bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
		#print("D:")
		#print(D.T)
		alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
		bestStump['alpha'] = alpha  
		weakClassArr.append(bestStump)				  #store Stump Params in Array
		#print("classEst: ")
		#print(classEst.T)
		expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
		D = multiply(D,exp(expon))							  #Calc New D for next iteration
		D = D/D.sum()
		#calc training error of all classifiers, if this is 0 quit for loop early (use break)
		aggClassEst += alpha*classEst
		#print("aggClassEst: ")
		#print(aggClassEst.T)
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
		errorRate = aggErrors.sum()/m
		#print("total error: %f" %errorRate)
		if errorRate == 0.0: break
	return weakClassArr#,aggClassEst

def adaClassify(datToClass,classifierArr):
	dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArr)):
		#pdb.set_trace()
		classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
								 classifierArr[i]['thresh'],\
								 classifierArr[i]['ineq'])#call stump classify
		aggClassEst += classifierArr[i]['alpha']*classEst
		#print(aggClassEst)
	return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
	import matplotlib.pyplot as plt
	cur = (1.0,1.0) #cursor
	ySum = 0.0 #variable to calculate AUC
	numPosClas = sum(array(classLabels)==1.0)
	yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
	sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	#loop through all the values, drawing a line segment at each point
	for index in sortedIndicies.tolist()[0]:
		if classLabels[index] == 1.0:
			delX = 0; delY = yStep;
		else:
			delX = xStep; delY = 0;
			ySum += cur[1]
		#draw line from cur to (cur[0]-delX,cur[1]-delY)
		ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
		cur = (cur[0]-delX,cur[1]-delY)
	ax.plot([0,1],[0,1],'b--')
	plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
	plt.title('ROC curve for AdaBoost horse colic detection system')
	ax.axis([0,1,0,1])
	plt.show()
	print("the Area Under the Curve is: %d" %(ySum*xStep))

def preDataArrange():
	print('Load and arrange skeleton train csv file.')
	arrangeCsv('/mnt/share/sample_train/sample_skeleton_train.csv', './skeleton_train.csv')
	print('Load and arrange skeleton test csv file.')
	arrangeCsv('/mnt/share/sample_test/sample_skeleton_test.csv', './skeleton_test.csv')
	

if __name__ == "__main__":
	#preDataArrange()
	#pdb.set_trace()
	print('Load skeleton feature file.')
	datArr,labelArr = loadSkeletonData('./skeleton_train.csv')
	#pdb.set_trace()
	print('Train adaboost model for skeleton data.')
	classifierArray = adaBoostTrainDS(datArr, labelArr, 40)
	print('Load skeleton test file.')
	datArr,labelArr = loadSkeletonData('./skeleton_test.csv')
	# judge classify result
	resultArr = adaClassify(datArr,classifierArray)
	err = 0; num = 0
	#num = shape(resultArr)[0]
	for i,data in enumerate(resultArr):
		#pdb.set_trace()
		if labelArr[i] == +1:
			num += 1
			if data != labelArr[i]: err += 1
	errRate = float(err) / float(num)
	print('ERROR Rate = %f' %errRate)
	#pdb.set_trace()
	