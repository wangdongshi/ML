# For the sake of simplicity, the meaning of variables' names is as follows:
# [x]   training data -- x with m x n dimension
# [y]   training data -- y with m column
# [a]   Lagrange multiplier -- alpha with m column
#import pdb
import time
from random import *
from numpy import *
from itertools import islice
import matplotlib.pyplot as plt

# const value
DIM = 28*28
	
class optStruct:
	'''
	prediction error estimation structure
	'''
	def __init__(self, x, y, C, t, k):
		self.x = x
		self.y = y
		self.C = C
		self.t = t
		self.m = shape(x)[0]
		self.a = mat(zeros((self.m,1)))
		self.b = 0
		self.Cache = mat(zeros((self.m,2)))
		self.K = mat(zeros((self.m,self.m)))
		for i in range(self.m):
			self.K[:,i] = kernelGauss(self.x, self.x[i,:], k)


class supportVector:
	'''
	two-classification's support vector information structure
	'''
	def __init__(self, xSV, kSV, b):
		self.xSV = xSV
		self.kSV = kSV
		self.b = b
		

def loadData(fName, RowStart, RowEnd):
	'''
	load data set (training or test) from csv file
	@param1 : file name
	@param2 : start line number of csv
	@param3 : end line number of csv
	@return1 : input data matrix
	@return2 : input data label
	'''
	x = []
	y = []
	fr = open(fName)
	for i in islice(fr, RowStart, RowEnd):
		data = i.strip().split(',')
		y.append(int(data[0]))
		line = []
		for i in range(1, DIM+1):
			line.append(float(data[i])/255.0)
		x.append(line)
	return x, y


def kernelGauss(X, A, k):
	'''
	RBF kernel function (k = sqrt(2)*sigma)
	Note : If each K(x1, x2) is calculate separately, the vector operation
	of the last step cannot be used, and the efficiency is greatly reduced.
	@param1 : transform matrix
	@param2 : operation vector
	@param3 : falling step of gauss function (k = sqrt(2)*sigma)
	@return1 : kernel function vector (column vector)
	'''
	m,n = shape(X)
	K = mat(zeros((m, 1))) # kernel function vector (column vector)
	for i in range(m):
		d = X[i,:] - A
		K[i] = d * d.T
	K = exp(K / (-1 * k ** 2))
	return K


def getEk(param, k):
	'''
	calculate prediction error
	( Ei = [Sum aj*yj*K(xj,xi)+b)] - yi )
	@param1 : operation structure
	@param2 : column number
	@return1 : prediction error Ek
	'''
	Ek = float(multiply(param.a, param.y).T*param.K[:,k] + param.b)
	Ek -= float(param.y[k]) # 7.105
	return Ek


def setEk(param, k):
	'''
	update prediction error
	@param1 : operation structure
	@param2 : column number
	'''
	Ek = getEk(param,k)
	param.Cache[k] = [1,Ek]


def selectJRand(i, m):
	'''
	select SMO inner loop alpha's index randomly
	@param1 : the first alpha's index
	@param2 : row of matrix
	@return1 : second alpha index
	'''
	j = i
	while(j==i) : j = int(random.uniform(0,m))
	return j


def selectJ(param, i, Ei):
	'''
	select SMO inner loop alpha's index
	@param1 : operation structure
	@param2 : the first alpha index
	@param3 : the first alpha's prediction error
	@return1 : second alpha index
	@return2 : second alpha's prediction error
	'''
	jMax = -1
	maxDeltaE = 0
	param.Cache[i]=[1, Ei] # calculate the Ei, so set the Ei flag
	aValid = nonzero(param.Cache[:,0].A)[0] # get the set  (E!=0)
	if(len(aValid) > 0): # select alpha j from the set (E!=0)
		for j in aValid:
			if j == i : continue
			Ej = getEk(param, j)
			deltaE = abs(Ei-Ej)
			if(deltaE > maxDeltaE):
				jMax = j
				maxDeltaE = deltaE
		return jMax, getEk(param,jMax)
	else: # select random alpha j
		j = selectJRand(i, param.m)
		Ej = getEk(param, j)
		return j, Ej


def setAlpha(ajNewUnc, H, L):
	'''
	update alpha j
	@param1 : alpha j (second alpha) new, unc
	@param2 : the right or top end of box
	@param2 : the left or bottom end of box
	@return1 : temporary alpha (new,unc)
	'''
	if ajNewUnc > H: ajNewUnc = H
	if ajNewUnc < L: ajNewUnc = L
	return ajNewUnc


def innerL(i, param):
	'''
	platt SMO inner loop
	@param1 : the first alpha index
	@param2 : operation structure
	@return1 : alpha j update OK or NG
	'''
	L = 0.0
	H = 0.0
	Ei = getEk(param, i)
	if ( (param.y[i] * Ei < -param.t) and (param.a[i] < param.C)) or \
		((param.y[i] * Ei > param.t) and (param.a[i] > 0) ): # outer loop index is right?
		# ##############################################
		# (Here is a problem.)
		# Random selection is better than heuristic selection of j.
		# But random selection must many of iteration.
		# Why?
		if param.m <= 50000 :
			j = selectJRand(i, param.m) # for test
			Ej = getEk(param, j) # for test
			#j,Ej = selectJ(param, i, Ei)
		else :
			j,Ej = selectJ(param, i, Ei)
		# ##############################################
		aiOld = param.a[i].copy()
		ajOld = param.a[j].copy()
		if (param.y[i] != param.y[j]): # Graph 7.8 left
			L = max(0, param.a[j] - param.a[i])
			H = min(param.C, param.C + param.a[j] - param.a[i])
		else: # Graph 7.8 right
			L = max(0, param.a[j] + param.a[i] - param.C)
			H = min(param.C, param.a[j] + param.a[i])
		if L == H:
			#print("L == H")
			return 0 # can not change the alpha pair
		eta = param.K[i,i] + param.K[j,j] - 2.0 * param.K[i,j] # 7.107
		if eta <= 0:
			#print("eta <= 0")
			return 0
		param.a[j] += param.y[j] * (Ei - Ej) / eta # 7.106
		param.a[j] = setAlpha(param.a[j],H,L)
		setEk(param, j)
		if (abs(param.a[j] - ajOld) < 0.00001): # the moving of j is too short
			#print("alphaj isn't moving enough. a%d = %f" % (j, param.a[j]))
			return 0
		else : 
			param.a[i] += param.y[j]*param.y[i]*(ajOld - param.a[j]) # 7.109
			setEk(param, i)
			b1 = param.b - Ei - param.y[i]*(param.a[i]-aiOld)*param.K[i,i] - \
					param.y[j]*(param.a[j]-ajOld)*param.K[i,j] # 7.115
			b2 = param.b - Ej - param.y[i]*(param.a[i]-aiOld)*param.K[i,j] - \
					param.y[j]*(param.a[j]-ajOld)*param.K[j,j] # 7.116
			# update b
			if (0 < param.a[i]) and (param.a[i] < param.C): param.b = b1
			elif (0 < param.a[j]) and (param.a[j] < param.C): param.b = b2
			else: param.b = (b1 + b2) / 2.0
			#print("alpha pair is updated! i=%d,j=%d,Ei=%f,Ej=%f,ai=%f,aj=%f,L=%f,H=%f" % (i, j, Ei, Ej, param.a[i], param.a[j], L, H))
			return 1
	else:
		#print("the Ei is bigger than tolerance. E%d = %f" % (i, Ei))
		return 0


def PlattSMO(x, y, C, t, maxIter=100000, k=10.0):
	'''
	platt SMO main function
	@param1 : training data
	@param2 : training data's label
	@param3 : relaxation variable
	@param4 : tolerance of Ek
	@param5 : falling step of gauss function (k = sqrt(2)*sigma)
	@return1 : b (split hyperplane intercept)
	@return2 : a (alpha -- Lagrange multiplier)
	'''
	param = optStruct(x, y, C, t, k)
	#print("x =")
	#print(param.x)
	#print("K =")
	#print(param.K)
	#print("C = %d" % param.C)
	#print("t = %d" % param.t)
	#print("m = %d" % param.m)
	#print("")
	
	cnt = 0
	cNum = 0 # alpha pair changed number of times
	isOutside = True
	while (cnt < maxIter) and ((cNum > 0) or (isOutside)):
		cNum = 0
		if isOutside: # the alpha of sample data is out of range (0, C)
			for i in range(param.m):
				if (innerL(i,param) == 1):
					cNum += 1
					#print("out (0, C), iter: %d i:%d, alpha pairs changed %d times" % (cnt,i,cNum))
			cnt += 1
		else: # the alpha of sample data is in range (0, C)
			insideSet = nonzero((param.a.A > 0) * (param.a.A < C))[0]
			for i in insideSet:
				if (innerL(i,param) == 1):
					cNum += 1
					#print("in  (0, C), iter: %d i:%d, alpha pairs changed %d times" % (cnt,i,cNum))
			cnt += 1
		if isOutside : isOutside = False
		elif (cNum == 0): isOutside = True
		#print("Iteration number is %d. (time : %s)" % (cnt, time.ctime()))
	return param.b, param.a


def twoClassify(num=1000, target=0, begin=1, C=100, t=0.00001, k=10.0):
	'''
	make two classification between "Target" and other's number.
	@param1 : training data number
	@param2 : target digital
	@param3 : the first row of training data
	@param4 : relaxation variable
	@param5 : tolerance of Ex
	@param6 : falling step of gauss function (k = sqrt(2)*sigma)
	@return1 : xSV (matrix of support vectors)
	@return2 : kSV (label and alpha's cross multiply of support vectors)
	@return3 : b (intercept of split hyperplane)
	@return4 : p (confidence rate)
	'''
	print ("Digit %d's training data number is %d. (time : %s)" % (target, num, time.ctime()))
	
	# load MNIST data
	data,label = loadData('16.MNIST.train.csv', begin, num+1)
	for i in range(0, num) :
		if label[i] == target : label[i] = 1
		else : label[i] = -1
	x = mat(data)
	y = mat(label).transpose()
	
	# execute PLATT SMO method
	b,a = PlattSMO(x, y, C, t, 100000, k) # C=100 k=10 is important!
	
	# set support parameter
	sv = nonzero(a.A>0)[0]
	xSV = x[sv] # get matrix of support vectors (alpha>0)
	ySV = y[sv] # get labels of support vectors (alpha>0)
	aSV = a[sv] # get alphas of support vectors (alpha>0)
	kSV = multiply(ySV, aSV)
	print ("Digit %d have %d support vectors." % (target, shape(ySV)[0]))
	
	return xSV, kSV, b


def multiClassify(num=1000, C=100, t=0.00001, k=10.0):
	'''
	make ten classification from '0' to '9' by OVR
	@param1 : training data number
	@param2 : relaxation variable
	@param3 : tolerance of Ex
	@param4 : falling step of gauss function (k = sqrt(2)*sigma)
	@return1 : support vector information from 0 to 9
	'''
	sv = {}
	
	# make 10 two-classification
	for i in range(10):
		xSV, kSV, b = twoClassify(num, i, 1, C, t, k)
		sv[i] = supportVector(xSV, kSV, b)
	print ("Multi-classification constructed. (time : %s)" % time.ctime())
	return sv

	
def testDraw(num=1000, C=100, t=0.00001, k=10.0):
	# load MNIST data
	data,label = loadData('16.MNIST.train.csv', 1, num+1)
	for i in range(0, num) :
		if label[i] == 0 : label[i] = 1
		else : label[i] = -1
	x = mat(data)
	y = mat(label).transpose()

	# draw figure(No.) for test
	X = []
	Y = []
	C = []
	index = int(random.uniform(0,num))
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i in range (28):
		for j in range (28):
			X.append(int(28-i))
			Y.append(int(j%28))
			color = int(data[index][i+28*j]*255.0)
			color = (hex(color))[2:]
			if len(color) < 2 : color += '0'
			color = '#' + color + color + color
			C.append(color)
	ax.scatter(X, Y, c=C, marker='s')
	plt.legend('x1')
	plt.savefig('test.png')
	return
	

if __name__ == '__main__':
	testDraw(num=1000)
