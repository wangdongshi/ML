# https://github.com/murtazakhan28/MNIST-dataset-classification-using-neural-network-in-python-and-numpy

#import pdb
#import pickle
from os import listdir
import numpy as np
import random
import math
import time
from itertools import islice

DIM = 28*28

def loadData(fName, RowStart, RowEnd):
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

def shuffle_data(x, y):
	temp = list(zip(x, y))
	random.shuffle(temp)
	return zip(*temp)

class Layer:
	def __init__(self, inputDim, actFunction, n):
		# take care of the following setting. 
		# do not set all coefficient to zero.
		self.weights = np.random.randn(n, inputDim)
		self.biases = np.random.randn(n, 1)
		self.gradients = np.random.rand(n, inputDim)
		self.biasGradients = np.random.rand(n, inputDim)
		self.neurons = n
		self.error = np.zeros((n, 1))
		self.weightedSums = np.zeros((n, 1))
		self.activations = np.zeros((n, 1))
		self.activationFunction = actFunction
			
def crossEntropyLoss(modelOutput, actualTarget):
	activationLog = np.log(modelOutput)
	error = np.multiply(-1, np.multiply(actualTarget, activationLog))
	return np.sum(error)
			
def initializeModel(numberOfLayers, inputDim, neurons):
	model = []
	model.append(Layer(inputDim, "sigmoid", inputDim))
	for i in range(0, numberOfLayers - 1):
		model.append(Layer(model[-1].neurons, "sigmoid", neurons[i]))
	model.append(Layer(model[-1].neurons, "softmax", neurons[-1]))
	return model

def sigmoid(x):
	temp = np.zeros((len(x), 1))
	for i in range(0, len(x)): # Avoiding overflow
		if x[i] > 21: temp[i] = 21
		elif x[i] < -709: temp[i] = -709
		else: temp[i] = x[i]
	temp = np.multiply(temp, -1)
	return np.divide(1, np.add(1, np.exp(temp)))

def sigmoidGradient(activations):
	return np.multiply(activations, np.subtract(1, activations))
	
def softmax(logit):
	exponent = np.exp(logit)
	exponentSum = np.sum(exponent)
	return exponent / exponentSum

def softmaxLossGradient(modelOutput, actualTarget):
	return np.subtract(modelOutput, actualTarget)

def forwardPropagation(model, inputSample):
	model[0].activations = inputSample
	for i in range(1, len(model)):
		model[i].weightedSums = np.dot(model[i].weights, model[i - 1].activations) + model[i].biases
		if model[i].activationFunction == "sigmoid":
			model[i].activations = sigmoid(model[i].weightedSums)
		elif model[i].activationFunction == "softmax":
			model[i].activations = softmax(model[i].weightedSums)
	return model[-1].activations

def lossGradient(modelOutput, actualTarget):
	return softmaxLossGradient(modelOutput, actualTarget)

def backPropagation(model, loss, inputSample):
	model[0].activations = inputSample
	model[-1].error = loss
	
	for i in range(len(model) - 1, 0, -1):
		model[i].gradients = np.multiply(model[i - 1].activations.transpose(), model[i].error)
		model[i].biasGradients = model[i].error
		model[i - 1].error = np.multiply(np.dot(model[i].weights.transpose(), model[i].error), sigmoidGradient(model[i - 1].activations))
	return model

def weightUpdate(model, lr):
	for i in range(1, len(model)):
		model[i].weights = np.subtract(model[i].weights, np.multiply(lr, model[i].gradients))
		model[i].biases = np.subtract(model[i].biases, np.multiply(lr, model[i].biasGradients))
	return model
		
def trainModel(model, trainingData, epochs, learningRate):
	print("Training Started!")
	previousError = 0
	for i in range(0, epochs):
		error = 0
		print("Epoch :", i)
		startTime = time.time()
		for j in range(0, len(trainingData[0])):
			modelOutput = forwardPropagation(model, trainingData[0][j])
			error = error + crossEntropyLoss(modelOutput, trainingData[1][j])
			loss = lossGradient(modelOutput, trainingData[1][j])
			model = backPropagation(model, loss, trainingData[0][j])
			model = weightUpdate(model, learningRate)
		endTime = time.time()
		error = error / len(trainingData[0])
		deltaError = error - previousError
		previousError = error
		print("Error:", error)
		#print("Delta Error:", deltaError)
		#print("Time Taken :", str(endTime - startTime))
		print("---------------------------------")
	print("Training Finished!")
	return model
			
if __name__ == '__main__':

	# Load data
	train_set_x, train_set_y = loadData('../16.MNIST.train.csv', 1, 3001)
	test_set_x,  test_set_y  = loadData('../16.MNIST.train.csv', 37001, 42001)

	# Reshaping the 28 x 28 images into vectors of 784 x 1
	x_train, x_test = [], []
	for i in range(0, len(train_set_x)):
		x_train.append(np.reshape(train_set_x[i], (784, 1)))
	for j in range(0, len(test_set_x)):
		x_test.append(np.reshape(test_set_x[j], (784, 1)))

	x_train, train_set_y = shuffle_data(x_train, train_set_y)

	# Converting labels into One Hot notation
	y_train = []
	for i in range(0, len(train_set_y)):
		y_train.append(np.zeros((10, 1)))
		y_train[-1][train_set_y[i]] = 1
	y_test = []
	for i in range(0, len(test_set_y)):
		y_test.append(np.zeros((10, 1)))
		y_test[-1][test_set_y[i]] = 1
	
	trainingData = (x_train, y_train)
	testingData  = (x_test, y_test)

	model = initializeModel(2, 784, [30, 10])
	#model = initializeModel(1, 784, [10])
	model = trainModel(model, trainingData, 500, 0.1)

	#pickle.dump(model, open("MNIST_NeuralNetwork.nnt", 'wb')) # Saving the trained model using pickle

	#model = pickle.load(open("MNIST_NeuralNetwork.nnt", 'rb'))
	print("Test Started!")
	testingData = (x_test, y_test)
	tp = np.zeros((10, 1))
	fn = np.zeros((10, 1))
	for i in range(0, len(testingData[0])):
		yVector = forwardPropagation(model, testingData[0][i])
		if yVector.argmax() == testingData[1][i].argmax():
			#print("No.%d is OK. It's label is %d." % (i, testingData[1][i].argmax()))
			tp[testingData[1][i].argmax()] += 1
		else:
			#print("No.%d is NG. It's label is %d." % (i, testingData[1][i].argmax()))
			fn[testingData[1][i].argmax()] += 1
		#print("Predication is :")
		#print(yVector)
			
	for i in range(0, 10):
		print(i, ": OK:", tp[i], "--- NG:", fn[i], "--- Accuracy:", str((tp[i] * 100) / (tp[i] + fn[i])), "%")
	print("Test Finished!")
