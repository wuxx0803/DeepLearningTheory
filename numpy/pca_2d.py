from pylab import *

# Load data
x = loadtxt('pcaData.txt')

## Plot the data
#scatter(x[0,:], x[1,:])
#title('Raw Data')
#show()

# Implement pca to obtain matrix U
sigma = dot(x, transpose(x))/ shape(x)[1]
U,S,V = svd(sigma)
#k = shape(U)[1]
k = 1
epsilon = 1e-6

xRot = dot(transpose(U), x)
xTilde = dot(U[:,0:k], dot(transpose(U[:,0:k]), x))
xPCAWhite = dot(dot(diag(1/sqrt(S + epsilon)), transpose(U)), x)
xZCAWhite = dot(dot(dot(U, diag(1/sqrt(S + epsilon))),
                transpose(U)), x)


# Plot the U vector
subplot(321)
plot([0, U[0,0]], [0, U[1,0]])
plot([0, U[0,1]], [0, U[1,1]])
scatter(x[0,:],x[1,:])
title('Eigenvectors of Covariance')
show()

## Plot the rotated data
subplot(322)
scatter(xRot[0,:], xRot[1,:])
title('xRot')
show()

subplot(323)
scatter(xTilde[0,:], xTilde[1,:])
title('xTilde')
show()

subplot(324)
scatter(xPCAWhite[0,:], xPCAWhite[1,:])
title('xPCAWhite')
show()

subplot(325)
scatter(xZCAWhite[0,:], xZCAWhite[1,:])
title('xZCAWhite')
show()

subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5,
        hspace=0.5)
