# For the sake of simplicity, the meaning of variables' names is as follows:
# [x]   training data -- x with m x n dimension
# [y]   training data -- y with m column
# [a]   Lagrange multiplier -- alpha with m column
#import pdb
import time
from random import *
from numpy import *
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
		# layer number of NN
		self.num_layers = len(sizes)
		self.sizes = sizes
		# cerate neurons' bias value of all of layer (0 - 1)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		# cerate every weight value of all coefficient (0 - 1)
		self.weights = [np.random.randn(y, x)
						for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		"""
		feed forward for every neuron's value 
		:param a		: input values of neurons
		:return			: output values of neurons
		"""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta,
			test_data=None):
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
		for j in xrange(epochs):
			# make training data set to random set
			random.shuffle(training_data)
			# divide training set by minimum sample number
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				# update w and b by every minimum sample
				self.update_mini_batch(mini_batch, eta)
			# print NN's correct rate after every test
			if test_data:
				print "Epoch {0}: {1} / {2}".format(
					j, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(j)

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
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		# update w and b with the accumulated partial derivatives
		# Note: divide the length of small sample for eta
		self.weights = [w-(eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb
					   for b, nb in zip(self.biases, nabla_b)]

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
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# get sigma value(after sigmoid transfer)
		delta = self.cost_derivative(activations[-1], y) * \
			sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		# multiply the output value of the previous layer
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in xrange(2, self.num_layers):
			# update from the last l-th layer
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		# get predicted result
		test_results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
		# return the number of correctly predict
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""
		loss function
		:param output_activations	: output activations
		:param y					: label value
		:return						: loss value
		"""
		return (output_activations-y)

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


if __name__ == '__main__':
	# test NN and softmax
	print ("Test NN and softmax regression by MNIST set.")
	print ()
