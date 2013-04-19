from pylab import *
from scipy.io import loadmat

def display_network(A, opt_normalize=True, cols=None):
  # rescale input
  print "HERE!"
  A = A - mean(A)
  # Compute the rows and columns
  L, M = shape(A)
  sz = sqrt(L)
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

  #arr = -ones(buf + m * (sz + buf), buf + n * (sz + buf))

  #k = 1
  print "m: " + str(m)
  print "n: " + str(n)
  print "FOO!"
  for i in range(int(m)):
    for j in range(int(n)):
      if k > M:
        continue
      clim = max(abs(A[:,k]))
      base1 = buf + (i - 1) * (sz + buf)
      base2 = buf + (j-1) * (sz + buf)
      if opt_normalize:
        array[base1:base1+sz,base2:base2+sz] = (reshape(A[:,k],
                                                (sz,sz), order='F')/ clim)
      else:
        array[base1:base1+sz, base2:base2+sz] = (reshape(A[:,k],
                                                 (sz,sz), order='F')/max(abs(A)))
      k = k + 1
  plt.imshow(array)
  plt.show()
