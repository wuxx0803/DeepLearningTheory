from pylab import *
from scipy.io import loadmat

def sampleImages():
  # define some constant
  patchsize = 8
  numpatches = 10000

  # Load matlab matrix
  d = loadmat('IMAGES.mat')
  IMAGES = d['IMAGES']

  patches = zeros(patchsize * patchsize, numpatches)

  # Sample the patches

  patches = normalizeData(patches)

  return patches

def normalizeData(patches):
  patches = patches - mean(patches)

  # Truncate to +/- 3 standard deviations and scale to -1 to 1
  pstd = 3 * std(patches.flatten())
  patches = max(min(patches, pstd), -pstd) / pstd;

  patches = (patches + 1) * 0.4  + 1
  return patches

def checkNumericalGradient():
  x = array([4, 10])

  # Check numerical gradient on a simple function
  value, grad = simpleQuadraticFunction(x)
  numgrad = computeNumericalgradient(simpleQuadraticFunction, x)
  print transpose(array([numgrad, grad]))

  diff = norm(numgrad - graad)/norm(numgrad+grad)
  print str(diff)


def simpleQuadraticFunction(x):
  value = x[0]**2 + 3 * x[0] * x[1]
  grad = zeros((2,1))
  grad[0] = 2 * x[0] + 3 * x[1]
  grad[1] = 3 * x[0]
  return value, grad

def computeNumericalGradient(J, theta):
  pass

def display_network(A, opt_normalize, opt_graycolor, cols, opt_colmajor):
  pass

def initializeParameters(hiddenSize, visibleSize):
  # I have no idea why this is used...
  r = sqrt(6) / sqrt(hiddenSize + visibleSize + 1)

  W1 = rand(hiddenSize, visibleSize) * 2 * r - r
  W2 = rand(visibleSize, hiddenSize) * 2 * r - r

  b1 = zeros((hiddenSize, 1))
  b2 = zeros((visibleSize, 1))

  theta = concatenate((W1.flatten(), W2.flatten(), b1, b2))
  return theta

def sparseAutoencoderCost(theta, visibleSize, hiddenSize, lmbda, sparsityParam,
    beta, data):
  pass

def sigmoid(x):
  pass

def train(hiddenSize):
  visibleSize = 8 * 8
  sparsityParam = 0.01

  lmbda = 0.0001
  beta = 3

  patches = sampleImages()

  theta = initializeParameters(hiddenSize, visibleSize)

  cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lmbda,
      sparsityParam, beta, patches)

  checkNumericalGradient()

  numGrad = computeNumericalGradient(lambda x: sparseAutoencoderCost(x,
    visibleSize, hiddenSize, lmbda, sparsityParam, beta, patches), theta)

  diff = norm(numgrad - grad)/norm(numgrad+grad)

  theta = initializeParameters(hiddenSize, visibleSize);

  optTheta, cost, d = fmin_l_bfgs_b(lambda x: sparseAutoencoderCost(x,
    visibleSize, hiddenSize, lmbda, sparsityParam, beta, patches), theta)

  W1 = reshape(optTheta[1:hiddenSize * visibleSize], hiddenSize, visibleSize)


  pass

if __name__ == "__main__":
  hiddenSize = 25
  train(25)
