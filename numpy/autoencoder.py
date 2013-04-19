from pylab import *
from scipy.io import loadmat
from scipy.optimize import fmin_l_bfgs_b

def display_network(A, opt_normalize=True, cols=None):
  # rescale input
  A = A - mean(A)
  # Compute the rows and columns
  L, M = shape(A)
  sz = int(sqrt(L))
  buf = 1
  if cols == None:
    if floor(sqrt(M))**2 != M:
      n = ceil(sqrt(M))
      while mod(M,n) != 0 and n < 1.2 * sqrt(M):
        n += 1
      m = ceil(M/n)
    else:
      n = sqrt(M)
      m = n
  else:
    n = cols
    m = ceil(M/n)
  m = int(m)
  n = int(n)

  arr = -ones((buf + m * (sz + buf), buf + n * (sz + buf)))
  k = 0
  for i in range(int(m)):
    for j in range(int(n)):
      if k >= M:
        continue
      clim = max(abs(A[:,k]))
      base1 = buf + i * (sz + buf)
      base2 = buf + j * (sz + buf)
      if opt_normalize:
        arr[base1:base1+sz,base2:base2+sz] = (reshape(A[:,k],
                                                (sz,sz), order='F')/ clim)
      else:
        arr[base1:base1+sz, base2:base2+sz] = (reshape(A[:,k],
                                                 (sz,sz), order='F')/max(abs(A)))
      k = k + 1
  plt.imshow(arr, cmap = cm.Greys_r)
  plt.show()
  return A

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

def sparseAutoencoderCostVectorized(theta, visibleSize, hiddenSize, eta,
    sparsityParam, beta, data):
  L = hiddenSize * visibleSize
  W1 = reshape(theta[:L],
      (hiddenSize, visibleSize), order = 'F')
  W2 = reshape(theta[L:2*L],
      (visibleSize, hiddenSize), order = 'F')
  b1 = reshape(theta[2*L:2*L + hiddenSize], (hiddenSize, 1), order='F')
  b2 = reshape(theta[2*L + hiddenSize:], (visibleSize, 1), order='F')

  # Initialize Parameters
  cost = 0
  W1grad = zeros(shape(W1))
  W2grad = zeros(shape(W2))
  b1grad = zeros(shape(b1))
  b2grad = zeros(shape(b2))
  rho = sparsityParam
  numExamples = shape(data)[1]

  # Forward propagation
  a2 = sigmoid(dot(W1, data) + tile(b1, (1, numExamples)))
  a3 = sigmoid(dot(W2, a2) + tile(b2, (1, numExamples)))

  # Compute average activation of all hidden neurons
  rhoHat = mean(a2, 1)
  sparseDelta = beta * ((1 - rho) / (1 - rhoHat) - rho / rhoHat)

  # Backward propagation
  diff = data - a3
  delta3 = -diff * a3 * (1 - a3)

  delta2 = ((dot(transpose(W2), delta3) +
            tile(reshape(sparseDelta, (hiddenSize, 1), order='F'),
              (1, numExamples)))
            * a2 * (1 - a2))

  # Square difference term for cost and gradient
  cost = trace(dot(diff, transpose(diff))) / (2 * numExamples)
  W1grad = dot(delta2, transpose(data)) / numExamples
  W2grad = dot(delta3 , transpose(a2)) / numExamples
  b1grad = sum(delta2, 1) / numExamples
  b2grad = sum(delta3, 1) / numExamples

  # Add regularization term to cost and gradient
  cost = cost + (eta / 2) * (norm(W1, 'fro')**2 + norm(W2, 'fro')**2)
  W1grad = W1grad + eta * W1
  W2grad = W2grad + eta * W2

  # Add sparsity term to the cost
  sparseCost = sum(rho * log(rho / rhoHat) + (1 - rho) *
      log((1-rho)/(1-rhoHat)))
  cost = cost + beta * sparseCost

  # convert to vector form
  grad = concatenate((W1grad.flatten('F'), W2grad.flatten('F'),
          b1grad.flatten('F'), b2grad.flatten('F')))
  return cost, grad

def sparseAutoencoderCost(theta, visibleSize, hiddenSize, eta, sparsityParam,
    beta, data):
  """ visibleSize: number of input units
      hiddenSize: number of hidden units
      eta: weight decay parameter
      sparsityParam: desired average activation for hidden units
      beta: weight for sparsity penalty term
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
    eta, beta, patches, vectorized, theta):

  #print "About to compute sparseAutoencoderCost"
  if vectorized:
    cost, grad = sparseAutoencoderCostVectorized(theta, visibleSize,
        hiddenSize, eta, sparsityParam, beta, patches)
  else:
    cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, eta,
        sparsityParam, beta, patches)

  print "About to computeNumericalGradient"
  numGrad = computeNumericalGradient(lambda x: sparseAutoencoderCost(x,
    visibleSize, hiddenSize, eta, sparsityParam, beta, patches)[0], theta)

  diff = norm(numGrad - grad,2)/norm(numGrad+grad,2)
  print str(diff)
  print 'Norm of the difference between numerical and analytical gradient (should be < 1e-9)'
  return numGrad, grad

def train(hiddenSize, visibleSize, sparsityParam, eta, beta, numPatches):
  # Set the parameters
  numIter = 400
  L = hiddenSize * visibleSize
  patches = sampleImages(numPatches)

  theta = initializeParameters(hiddenSize, visibleSize);
  optTheta, cost, d = fmin_l_bfgs_b(lambda x: sparseAutoencoderCostVectorized(x,
    visibleSize, hiddenSize, eta, sparsityParam, beta, patches), theta,
    maxfun = numIter)
  W1 = reshape(optTheta[0:L], (hiddenSize, visibleSize), order='F')
  W2 = reshape(optTheta[L:2*L], (hiddenSize, visibleSize), order='F')
  b1 = optTheta[2*L:2*L + hiddenSize]
  b2 = optTheta[2*L + hiddenSize:]
  return (W1,W2, b1,b2)

if __name__ == "__main__":
  hiddenSize = 25
  visibleSize = 8 * 8
  sparsityParam = 0.01
  eta = 0.0001
  beta = 3
  numPatches = 10000

  DEBUG = True
  if DEBUG:
    numPatches = 10
    patches = sampleImages(numPatches)
    theta = initializeParameters(hiddenSize, visibleSize)
    print "Vectorized Gradient Check:"
    numGradV, gradV = checkAutoencoderGradient(hiddenSize, visibleSize,
        sparsityParam, eta, beta, patches, True, theta)
    print "Standard Gradient Check:"
    numGrad, grad = checkAutoencoderGradient(hiddenSize, visibleSize,
        sparsityParam, eta, beta, patches, False, theta)
  else:
    numPatches = 10000
    W1, W2, b1, b2 = train(hiddenSize, visibleSize, sparsityParam, eta,
        beta, numPatches)
    A= display_network(transpose(W1))
