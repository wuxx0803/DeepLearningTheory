from pylab import *
# Code to perform softmax

def softmaxPredict(softmaxModel, data):
  pass

def softmaxTrain(inputSize, numClasses, lmbda, inputData, labels, maxIter):
  pass

def softmaxCost(theta, numClasses, inputSize, lmbda, data, labels);
  """ numClasses - number of label classes
      inputSize - the size N of input vector
      lmbda - weight decay parameter
      data - the N x M input matrix whose column d[:,i] corresponds to
             a single test set
      labels - a M x 1 matrix containing labels corresponding to input data
  """

  # Unroll the parameters from theta
  theta = reshape(theta, (numClasses, inputSize), order='F')

  numCases = shape(data)[1]

  pass


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
