# For the sake of simplicity, the meaning of variables' names is as follows:
# [x]   training data -- x with m x n dimension
# [y]   training data -- y with m column
# [a]   Lagrange multiplier -- alpha with m column
#import pdb
import time
import random
import numpy as np
from itertools import islice

# const value
DIM = 28*28


class Network(object):

	def __init__(self, sizes):
		"""
		:param sizes:	list type, save the number of neuron network
						for example, sizes = [2, 3, 2]
						it means there are 2 neurons in input layer,
						and 3 neurons in hidden layer, and 2 neurons in 
						output layer.
		"""
		# adapt to softmax transfer to last layer?
		self.softmax = False
		# layer number of NN
		self.num_layers = len(sizes)
		self.sizes = sizes
		# cerate neurons' bias value of all of layer (0 - 1)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		# cerate every weight value of all coefficient (0 - 1)
		self.weights = [np.random.randn(y, x)
						for x, y in list(zip(sizes[:-1], sizes[1:]))]

	def feedforward(self, a):
		"""
		feed forward for every neuron's value 
		:param a		: input values of neurons
		:return			: output values of neurons
		"""
		for b, w in list(zip(self.biases, self.weights)):
			a = sigmoid(np.dot(w, a) + b)
		# set last layer's active function to softmax
		if self.softmax : a = softmax(a)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta,
			softmax=False, test_data=None):
		"""
		stochastic gradient descent
		:param training_data	: input training data set
		:param epochs			: number of iterations
		:param mini_batch_size	: minimum sample number
		:param eta				: learning rate
		:param test_data		: test data set
		"""
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in list(range(epochs)):
			# make training data set to random set
			random.shuffle(training_data)
			# divide training set by minimum sample number
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in list(range(0, n, mini_batch_size))]
			for mini_batch in mini_batches:
				# update w and b by every minimum sample
				self.update_mini_batch(mini_batch, eta)
			# print NN's correct rate after every test
			if test_data:
				print ("Epoch {0}: {1} / {2}".format(
					j, self.evaluate(test_data), n_test))
			else:
				print ("Epoch {0} complete".format(j))

	def update_mini_batch(self, mini_batch, eta):
		"""
		Update w and b
		:param mini_batch	: the parts of sample
		:param eta			: learning rate
		"""
		# create 0 matrix by column and row number of biases and weights
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			# calculate the partial derivatives of w and b by each output 
			# y and each input x in the sample
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			# accumulate storage partial values delta_nabla_b and delta_nabla_w
			nabla_b = [nb+dnb for nb, dnb in list(zip(nabla_b, delta_nabla_b))]
			nabla_w = [nw+dnw for nw, dnw in list(zip(nabla_w, delta_nabla_w))]
		# update w and b with the accumulated partial derivatives
		# Note: divide the length of small sample for eta
		self.weights = [w-(eta/len(mini_batch))*nw
						for w, nw in list(zip(self.weights, nabla_w))]
		self.biases = [b-(eta/len(mini_batch))*nb
						for b, nb in list(zip(self.biases, nabla_b))]

	def backprop(self, x, y):
		"""
		back propagation (BP algorithm)
		:param x	: input value set
		:param y	: label value set
		:return		: calculated delta_nabla_b and delta_nabla_w
		"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# forward propagation
		activation = x
		# save the matrix of neurons' value before sigmoid transfer
		activations = [x]
		zs = []
		size = len(list(zip(self.biases, self.weights)))
		for i, (b, w) in enumerate(list(zip(self.biases, self.weights))):
		#for b, w in list(zip(self.biases, self.weights)):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			if self.softmax and i == size - 1 : activation = softmax(activation)
			activations.append(activation)
		if self.softmax :
			delta = self.cost_derivative(activations[-1], y)
		else :
		##########################################################
		##   the following process is made by sigmoid
		##########################################################
		# get delta value(after sigmoid transfer)
			delta = self.cost_derivative(activations[-1], y) * \
				sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		# multiply the output value of the previous layer
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in list(range(2, self.num_layers)):
			# update from the last l-th layer
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		##########################################################
		
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		# get predicted result
		#test_results = [(self.feedforward(x), np.argmax(self.feedforward(x)), y)
		test_results = [(self.feedforward(x), np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
		#print (test_results)

		# return the number of correctly predict
		return sum(int(x == y) for (array, x, y) in test_results)

	def cost_derivative(self, activations, y):
		"""
		loss function (not only sigmoid but also softmax)
		:param activations	: output activations
		:param y			: label value
		:return				: loss value
		"""
		return (activations - y)

#### Utility functions
def sigmoid(z):
	"""
	calculate sigmoid function's value
	:param z	: input value
	:return		: sigmoid transfer value
	"""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""
	calculate sigmoid function's derivative
	:param z	: input value
	:return		: sigmoid derivative's value
	"""
	return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
	"""
	calculate softmax function's value
	:param z	: input vector
	:return		: softmax transfer value
	"""
	return np.exp(z)/np.sum(np.exp(z))


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
	xMat = [np.reshape(s, (784, 1)) for s in x]
	yMat = [vectorized_result(s) for s in y]
	return list(zip(xMat, y))

def vectorized_result(j):
	'''
	transfer y to 10-dimensinal unit vector with a
	1.0 in the jth position and 0.0 elsewhere.
	@param1 : y's positon
	@return1 : 10-dimensinal vector
	'''
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e


if __name__ == '__main__':
	# test NN and softmax
	print ("Test NN and softmax regression by MNIST set.")

	# load data
	training_set = loadData('../16.MNIST.train.csv', 1, 1001)
	training_data = []
	for i in range(0,len(training_set)):
		data = [training_set[i][0], vectorized_result(training_set[i][1])]
		training_data.append(data)
	validation_data = loadData('../16.MNIST.train.csv', 10001, 11001)
	test_data = loadData('../16.MNIST.train.csv', 37001, 42001)
	#test_data = loadData('../16.MNIST.train.csv', 1, 1001)

	# test NN
	net = Network([784, 30, 10])
	net.SGD(training_data, 100, 10, 0.5, softmax=True, test_data = test_data)
	print ("Completed!")
