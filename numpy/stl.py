from pylab import *
from autoencoder import *

def feedForwardAutoencoder(theta, hiddenSize, visibleSize, data):
  L = hiddenSize * visibleSize
  W1 = reshape(theta[1:L], (hiddenSize, visibleSize), order='F')
  b1 = theta[2*L:2*L + hiddenSize]
  activation = sigmoid(dot(W1, data) + tile(b1, (1, shape(data)[1])))
  return activation

if __name__ == "__main__":
  inputSize = 28 * 28
  numLabels = 5
  hiddenSize = 200
  eta = 3e-3
  beta = 3
  maxIter = 400

  L = hiddenSize * visibleSize
  mnistData = loadMNISTImages('train-images-idx3-ubyte')
  mnistLabels = loadMNISTLabels('train-labels-idx1-ubyte')

  labeledSet = mnistLabels[logical_and(mnistLabels >= 0, mnistLabels <=4)]
  unlabeledSet = mnistLabels[mnistLabels >= 5]

  numTrain = around(size(labeledSet)/2)
  trainSet = labeledSet[:numTrain]
  testSet = labeledSet[numTrain:end]

  unlabeledData = mnistData[:,unlabeledSet]

  trainData = mnistData[:, trainSet]
  trainLabels = transpose(mnistLabels[trainSet])

  testData = mnistData[:, testSet]
  testLabels = transpose(minstLabels[testSet])

  print '# examples in unlabeled set: %d\n' % shape(unlabeledData)[1]
  print '# examples in supervised training set: %d\n\n' % shape(trainData)[1]
  print '# examples in supervised testing set: %d\n\n' % shape(testData][1]

  #  Randomly initialize the parameters
  theta = initializeParameters(hiddenSize, inputSize)

  # Add call to lbfgs...

  # Visualize Weights
  W1 = reshape(optTheta[:L], (hiddenSize, inputSize), order='F')
  display_network(transpose(W1))

  # Extract the features
  trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize,
      trainData)

  testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, testData)

  # Train the softmax classifier
  optTheta = softmaxTrain(hiddenSize, 5, 1e-4, trainFeatures, trainLabels);

  pred = softmaxPredict(softmaxModel, testFeatures);

  ## ----------------------------------------------------

  # Classification Score
  print 'Test Accuracy: %f%%\n' % 100*mean(pred == testLabels)
