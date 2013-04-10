from pylab import *
# Code to perform softmax

def h(x, theta):
  """ x - a N x 1 vector corresponding to a single data point
      theta - a N x k vector of parameters
  """
  v = exp(dot(transpose(theta), x))
  Z = sum(v)
  return (1/float(Z)) * v


def softmaxPredict(theta, data):
  """ data - the N x M input matrix
      produces preduction matrix pred, where pred(i) = argmax_c P(y(c) | x(i))
  """
  K = shape(theta)[0]
  N = shape(theta)[1]
  M = shape(data)[1]

  costMatrix = exp(dot(theta, data))
  columnSums = sum(costMatrix, axis=0)
  normalizationMatrix = tile(columnSums, (K, 1))
  probMatrix = costMatrix / normalizationMatrix

  pred = argmax(probMatrix, axis=0)
  return pred

def softmaxTrain(inputSize, numClasses, lmbda, inputData, labels, maxIter):
  numCases = shape(labels)[0]

  # Label some constants
  K = numClasses
  N = inputSize
  M = numCases
  theta = 0.005 * randn(K * N, 1)
  optTheta, cost, d = fmin_l_bfgs_b(lambda x: softmaxCost(x, numClasses,
    inputSize, lmbda, data, labels), theta)

  optTheta = reshape(optTheta, K, N, order='F')

  return optTheta

def softmaxCost(theta, numClasses, inputSize, lmbda, data, labels);
  """ numClasses - K number of label classes
      inputSize - the size N of input vector
      lmbda - weight decay parameter
      data - the N x M input matrix whose column d[:,i] corresponds to
             a single test set
      labels - a M x 1 matrix containing labels corresponding to input data
  """
  numCases = shape(data)[1]

  # Label some constants
  K = numClasses
  N = inputSize
  M = numCases

  # Unroll the parameters from theta
  theta = reshape(theta, (K, N), order='F')

  groundTruth = zeros(K, M)
  groundTruth[labels,arange(M)] = 1

  costMatrix = exp(dot(theta, data))
  columnSums = sum(costMatrix, axis=0)
  normalizationMatrix = tile(columnSums, (K, 1))
  cost = - (1.0 /M) * sum(groundTruth * log(costMatrix / normalizationMatrix))

  coeffMatrix = costMatrix / normalizationMatrix

  # Inflate the matrices to 3d and tile them
  coeffMatrix.resize((K,M,1))
  coeffMatrix = tile(coeffMatrix,(1,1,N))

  dataMatrix = transpose(data)
  dataMatrix.resize((1,M,N))
  dataMatrix = tile(dataMatrix,(K,1,1))

  grad = -(1.0/M) * sum(dataMatrix * coeffMatrix, axis = 1)
  return cost, grad

#def softmaxGradient(theta, numClasses, lmbda, data, labels):
#  thetagrad = zeros((numClasses, inputSize))
#  # Unroll the parameters from theta
#  theta = reshape(theta, (numClasses, inputSize), order='F')
#  numCases = shape(data)[1]
#
#  # Label some constants
#  K = numClasses
#  N = inputSize
#  M = numCases
#
#  # Calculate the indicator matrix
#  groundTruth = zeros(K, M)
#  groundTruth[labels,arange(M)] = 1
#
#  # Calculate the cost matrix
#  costMatrix = exp(dot(theta, data))
#  columnSums = sum(costMatrix, axis=0)
#  normalizationMatrix = tile(columnSums, (K, 1))
#  coeffMatrix = costMatrix / normalizationMatrix
#
#  # Inflate the matrices to 3d and tile them
#  coeffMatrix.resize((K,M,1))
#  coeffMatrix = tile(coeffMatrix,(1,1,N))
#
#  dataMatrix = transpose(data)
#  dataMatrix.resize((1,M,N))
#  dataMatrix = tile(dataMatrix,(K,1,1))
#
#  grad = -(1.0/M) * sum(dataMatrix * coeffMatrix, axis = 1)
#  return grad

if __name__ = "__main__":

  # Initialize Parameters
  inputSize = 28 * 28
  numClasses = 10
  lmbda = 1e-4

  # Load the Data
  images =  loadMNISTImages('train-images-idx3-ubyte')
  labels =  loadMNISTLabels('train-labels-idx3-ubyte')

  # Relabel 0 to 10
  labels[labels==0] = 10

  inputData = images

  # If debugging, reduce the input size
  DEBUG = True
  if DEBUG:
      inputSize = 8
      inputData = randn(8,100)
      labels = randi(10,100,1)

  # Randomly initialize theta
  theta = 0.005 * randn(numClasses * inputSize, 1)

  # Find gradient and current softmax cost
  cost, grad = softmaxCost(theta, numClasses, inputSize, lmbda, inputData,
          labels)

  # If in debug mode, check the numerical gradient

  if DEBUG:
      numGrad = computeNumericalGradient(lambda x: softmaxCost(x,
          numClasses, inputSize, lmbda, inputData, labels), theta)

      # Compute the numeric gradient to the analytic gradient
      diff = norm(numGrad - grad)/norm(numGrad + grad)
      print("Diff = " + str(diff))

  # Learn Parameters
  maxIter = 100
  softmaxModel = softmaxTrain(inputSize, numClasses, lmbda, inputData, labels,
          options)

  # Test on True Dataset
  images = loadMNISTImages('t10k-images-idx3-ubyte')
  labels = loadMNISTLabels('t10k-labels-idx1-ubyte')
  labels(labels==0) = 10

  inputData = images
  pred = softmaxPredict(softmaxModel, inputData)

  acc = mean(labels == pred)

  print("Accuracy: " + str(acc * 100))

if __name__ == "__main__":
  print "nothing yet!"
