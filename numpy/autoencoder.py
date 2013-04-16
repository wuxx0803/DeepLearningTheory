from pylab import *
from scipy.io import loadmat
from scipy.optimize import fmin_l_bfgs_b
from display_network import *

def sampleImages(numPatches):
  # define some constant
  patchsize = 8

  # Load matlab matrix
  d = loadmat('IMAGES.mat')
  IMAGES = d['IMAGES']

  patches = zeros((patchsize ** 2, numPatches))
  xSize, ySize, numImages = shape(IMAGES)

  # Sample the patches
  iVals = randint(numImages, size=numPatches)
  xVals = randint(xSize - patchsize, size=numPatches)
  yVals = randint(ySize - patchsize, size=numPatches)

  for n in range(numPatches):
    i,x,y = iVals[n], xVals[n], yVals[n]
    patch = IMAGES[y:y+patchsize,x:x+patchsize, i]
    patches[:,n] = reshape(patch, patchsize ** 2, order='F')

  patches = normalizeData(patches)
  return patches

def normalizeData(patches):
  patches = patches - mean(patches)

  # Truncate to +/- 3 standard deviations and scale to -1 to 1
  pstd = 3 * std(patches)
  patches = maximum(minimum(patches, pstd), -pstd) / pstd;

  patches = (patches + 1) * 0.4  + 1
  return patches

def checkNumericalGradient():
  x = array([4.0, 10])

  # Check numerical gradient on a simple function
  value, grad = simpleQuadraticFunction(x)
  numgrad = computeNumericalGradient(lambda y: simpleQuadraticFunction(y)[0], x)
  print transpose(array([numgrad, grad]))

  diff = norm(numgrad - grad,2)/norm(numgrad+grad,2)
  print str(diff)
  print 'Norm of the difference between numerical and analytical gradient (should be < 1e-9)'
  return numgrad, grad


def simpleQuadraticFunction(x):
  value = x[0]**2 + 3 * x[0] * x[1]
  grad = zeros(2)
  grad[0] = 2 * x[0] + 3 * x[1]
  grad[1] = 3 * x[0]
  return value, grad

def computeNumericalGradient(J, theta):
  """ theta: parameter vector
      J: function that outputs a real number (i.e., y = J(theta))
  """
  EPSILON = 1e-4
  numGrad = zeros(shape(theta))
  for i in range(len(theta)):
    if i % 100 == 0:
      print "J = " + str(i) + "/" + str(len(theta))

    # Compute the thetas without copying the vector
    theta[i] = theta[i] + EPSILON
    JthetaP = J(theta)
    theta[i] = theta[i] - 2 * EPSILON
    JthetaN = J(theta)
    theta[i] = theta[i] + EPSILON

    numGrad[i] = (JthetaP - JthetaN)/(2 * EPSILON)
  return numGrad

def initializeParameters(hiddenSize, visibleSize):
  r = sqrt(6) / sqrt(hiddenSize + visibleSize + 1)
  W1 = rand(hiddenSize, visibleSize) * 2 * r - r
  W2 = rand(visibleSize, hiddenSize) * 2 * r - r
  b1 = zeros((hiddenSize, 1))
  b2 = zeros((visibleSize, 1))

  theta = concatenate((W1.flatten('F'), W2.flatten('F'),
                      b1.flatten('F'), b2.flatten('F')))
  return theta

def sparseAutoencoderCost(theta, visibleSize, hiddenSize, eta, sparsityParam,
    beta, data):
  """ visibleSize: number of input units
      hiddenSize: number of hidden units
      eta: weight decay parameter
      sparsityParam: desired average activation for hidden units
      beta: weight for sparsity penalty term
      TODO: vectorize this code
  """
  L = hiddenSize * visibleSize
  W1 = reshape(theta[:L],
      (hiddenSize, visibleSize), order = 'F')
  W2 = reshape(theta[L:2*L],
      (visibleSize, hiddenSize), order = 'F')
  b1 = theta[2*L:2*L + hiddenSize]
  b2 = theta[2*L + hiddenSize:]

  # Initialize Parameters
  cost = 0
  W1grad = zeros(shape(W1))
  W2grad = zeros(shape(W2))
  b1grad = zeros(shape(b1))
  b2grad = zeros(shape(b2))
  rho = sparsityParam
  numExamples = shape(data)[1]

  # Compute forward propagation for examples
  a2 = zeros((hiddenSize, numExamples))
  a3 = zeros((visibleSize, numExamples))
  for i in range(numExamples):
    x = data[:,i]
    a2[:,i] = sigmoid(dot(W1, x) + b1)
    a3[:,i] = sigmoid(dot(W2, a2[:,i]) + b2)

  rhoHat = mean(a2,1)
  sparseDelta = beta * ((1-rho)/(1 - rhoHat) - rho / rhoHat)

  # Compute backward propagation
  delta2 = zeros(shape(a2))
  delta3 = zeros(shape(a3))
  for i in range(numExamples):
    #if i % 1000 == 0:
    #  print "i = " + str(i)
    y = data[:,i]
    delta3[:,i] = -(y - a3[:,i]) * a3[:,i] * (1 - a3[:,i])
    delta2[:,i] = ((dot(transpose(W2), delta3[:,i]) + sparseDelta) *
                    a2[:,i] * (1 - a2[:,i]))
    cost += 0.5 * norm(y - a3[:,i])**2

    W1grad = W1grad + outer(delta2[:,i], data[:,i])
    W2grad = W2grad + outer(delta3[:,i], a2[:,i])
    b1grad = b1grad + delta2[:,i]
    b2grad = b2grad + delta3[:,i]

  cost = cost/numExamples
  W1grad = W1grad / numExamples
  W2grad = W2grad / numExamples
  b1grad = b1grad / numExamples
  b2grad = b2grad / numExamples

  # Add regularization cost
  cost += (eta/2) * (norm(W1, 'fro')**2 + norm(W2, 'fro')**2)
  W1grad = W1grad + eta * W1
  W2grad = W2grad + eta * W2

  # Add sparsity term to the cost
  sparseCost = sum(rho * log(rho/rhoHat) + (1 - rho) * log((1-rho)/(1-rhoHat)))
  cost = cost + beta * sparseCost

  grad = concatenate((W1grad.flatten('F'), W2grad.flatten('F'),
    b1grad.flatten('F'), b2grad.flatten('F')))
  return cost, grad

def sigmoid(x):
  return 1 / (1 + exp(-x))

def checkAutoencoderGradient(hiddenSize, visibleSize, sparsityParam,
    eta, beta, patches):
  theta = initializeParameters(hiddenSize, visibleSize)

  #print "About to compute sparseAutoencoderCost"
  cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, eta,
      sparsityParam, beta, patches)

  print "About to computeNumericalGradient"
  numGrad = computeNumericalGradient(lambda x: sparseAutoencoderCost(x,
    visibleSize, hiddenSize, eta, sparsityParam, beta, patches)[0], theta)

  diff = norm(numGrad - grad,2)/norm(numGrad+grad,2)
  print str(diff)
  print 'Norm of the difference between numerical and analytical gradient (should be < 1e-9)'

def train(numPatches):
  # Set the parameters
  visibleSize = 8 * 8
  hiddenSize = 25
  sparsityParam = 0.01
  eta = 0.0001
  beta = 3
  numIter = 400
  L = hiddenSize * visibleSize
  patches = sampleImages(numPatches)
  #checkAutoencoderGradient(hiddenSize, visibleSize, sparsityParam, eta, beta,
  #    patches)

  theta = initializeParameters(hiddenSize, visibleSize);
  optTheta, cost, d = fmin_l_bfgs_b(lambda x: sparseAutoencoderCost(x,
    visibleSize, hiddenSize, eta, sparsityParam, beta, patches), theta,
    maxfun = numIter)
  W1 = reshape(optTheta[0:L], (hiddenSize, visibleSize), order='F')
  W2 = reshape(optTheta[L:2*L], (hiddenSize, visibleSize), order='F')
  b1 = optTheta[2*L:2*L + hiddenSize]
  b2 = optTheta[2*L + hiddenSize:]
  return (W1,W2, b1,b2)

if __name__ == "__main__":
  #hiddenSize = 25
  #checkNumericalGradient()
  #patches = sampleImages()
  #numgrad, grad = checkNumericalGradient()
  numPatches = 10
  W1, W2, b1, b2 = train(numPatches)
  display_network(transpose(W1))
