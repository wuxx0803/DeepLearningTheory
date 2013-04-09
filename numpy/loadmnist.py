import scipy.io as io
from pylab import *

def loadMNISTImages(filename):
  f = open(filename, 'rb')

  # Verify Magic Number
  s = f.read(4)
  magic = int(s.encode('hex'),16)
  assert(magic == 2051)

  # Get Number of Images
  s = f.read(4)
  numImages = int(s.encode('hex'),16)
  s = f.read(4)
  numRows = int(s.encode('hex'),16)
  s = f.read(4)
  numCols = int(s.encode('hex'),16)

  # Get Data
  s = f.read()
  a = frombuffer(s, uint8)

  # Use 'F' to ensure that we read by column
  a = reshape(a, (numCols , numRows, numImages), order='F');
  images = transpose(a, (1, 0, 2))
  f.close()

  # Reshape to #pixels * #examples
  images  = reshape(a, (shape(images)[0] * shape(images)[1], numImages),
          order='F');
  images = double(images)/255
  return images

def loadMNISTLabels(filename):
  f = open(filename, 'rb')

  # Verify Magic Number
  s = f.read(4)
  magic = int(s.encode('hex'), 16)
  assert(magic == 2049)

  # Read Number Labels
  s = f.read(4)
  numLabels = int(s.encode('hex'), 16)

  # Get Data
  s = f.read()
  f.close()

  labels = frombuffer(s, uint8)
  assert(len(labels) == numLabels)
  return labels,s

if __name__ == "__main__":
    #images = loadMNISTImages('train-images-idx3-ubyte')
    labels,s = loadMNISTLabels('train-labels-idx1-ubyte')
