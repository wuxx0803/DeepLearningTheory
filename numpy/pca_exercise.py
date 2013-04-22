from pylab import *
from scipy.io import loadmat
from autoencoder import display_network

def sampleImagesRaw():
  d = loadmat('IMAGES_RAW.mat')
  IMAGES = d['IMAGESr']

  patchSize = 12
  numPatches = 10000

  patches = zeros((patchSize * patchSize, numPatches))

  p = 0
  for im in range(shape(IMAGES)[2]):
    # Sample patches
    numsamples = numPatches / shape(IMAGES)[2]
    for s in range(numsamples):
      y = randint(shape(IMAGES)[0] - patchSize)
      x = randint(shape(IMAGES)[1] - patchSize)
      sample = IMAGES[y:y+patchSize,x:x+patchSize,im]
      patches[:,p] = sample.flatten('F')
      p += 1
  return patches

def pca_gen():
  x = sampleImagesRaw()
  #randsel = randint(shape(x)[1],size=(200))
  #display_network(x[:,randsel])

  # zero mean the data
  avg = mean(x,0)
  x = x - avg

  # Implement PCA to obtain xRot
  sigma = dot(x, transpose(x))/ shape(x)[1]
  U,S,V = svd(sigma)
  xRot = dot(transpose(U), x)

  ## Check covariance for xRot
  #sigmaRot = dot(xRot, transpose(xRot))/ shape(xRot)[1]
  #plt.imshow(sigmaRot)
  #plt.show()

  # Choose k to retain 99% of the variance
  n = shape(S)[0]
  pctVar = 0.99
  k = calculateK(S, n, pctVar)
  #print "n: " + str(n)
  #print "pctVar: " + str(pctVar)
  #print "k: " + str(k)

  # Shrink the data to the correct dimension
  xTilde = dot(U[:,0:k], dot(transpose(U[:,0:k]), x))
  #randsel = randint(shape(xTilde)[1],size=(200))
  #display_network(xTilde[:,randsel])

  # Set regularization factor
  epsilon = 0.1
  # Calcualte xPCAWhite
  xPCAWhite = dot(dot(diag(1/sqrt(S + epsilon)), transpose(U)), xTilde)
  ## Check covariance for xPCAWhite
  #sigmaPCAWhite = dot(xPCAWhite, transpose(xPCAWhite))/ shape(xPCAWhite)[1]
  #plt.imshow(sigmaPCAWhite)
  #plt.show()

  # Calculate xZCAWhite
  xZCAWhite = dot(dot(dot(U, diag(1/sqrt(S + epsilon))),
                  transpose(U)), x)
  ## Check covariance for xZCAWhite
  #sigmaZCAWhite = dot(xZCAWhite, transpose(xZCAWhite))/ shape(xZCAWhite)[1]
  #plt.imshow(sigmaZCAWhite)
  #plt.show()
  randsel = randint(shape(xZCAWhite)[1],size=(200))
  display_network(xZCAWhite[:,randsel])

def calculateK(S, n, pctVar):
  sumEig = sum(S)
  sumSoFar = 0
  k = 0
  for i in range(n):
    sumSoFar += S[i]
    k += 1
    if sumSoFar/sumEig >= pctVar:
      break
  return k



if __name__ == "__main__":
  #patches = sampleImagesRaw()
  pca_gen()
