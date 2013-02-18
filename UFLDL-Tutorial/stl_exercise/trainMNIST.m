%% CS294A/CS294W Programming Assignment Starter Code

%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

visibleSize = 28*28;   % number of input units 
hiddenSize = 196;     % number of hidden units 
sparsityParam = 0.1;   % desired average activation of the hidden units.
lambda = 3e-3;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data

patches = loadMNISTImages('train-images-idx3-ubyte');
patches = patches(:, 1:10000);

%%======================================================================
%% STEP 2: Train the network

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCostVectorized(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

%%======================================================================
%% STEP 5: Visualization 

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 


