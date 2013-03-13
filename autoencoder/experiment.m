
load('mnist2k/MNIST_train2k.mat');
load('mnist2k/MNIST_test2k.mat');
trainData = X;
trainLabels = y;
testData = Xtest;
testLabels = ytest;
numClasses = max(trainLabels) - min(trainLabels) + 1;

inputSize = size(X, 1);
numTrainingExamples = size(X, 2);

% Draw examples from uniform random distribution
% data = rand(inputSize, numTrainingExamples);

% Draw data from gaussian distribution
% Sigma = [2, 1; 1, 3]; R = chol(Sigma);
% R = [sqrt(2), sqrt(2); -sqrt(2), sqrt(2)] * diag([1, 5]);
% data = randn(numTrainingExamples, 2) * R;
% data = data';

checkGradient = true;
hiddenSize = 200;
lambda = 3e-3;
beta = 3;
rho = 0.1;
f = @sigmoid;
phi = @phiL2;
softmaxLambda = 1e-4;

% Lets do PCA
% Mean subtraction:
% Xc = X - repmat(mean(X, 2), 1, numTrainingExamples);
% [U, S, V] = svd(Xc, 0);
% U = U(:, 1:hiddenSize);
% display_network(U);
% pause;

% Define the objective function
J = @(p) sparseAutoencoderCostVectorized(p, inputSize, hiddenSize, f, phi, lambda, rho, beta, trainData);

% Do numeric gradient checking
if checkGradient
    easyJ = @(p) sparseAutoencoderCostVectorized(p, inputSize, 3, f, phi, lambda, rho, beta, trainData(:, 1:10));
    easyTheta = initializeParameters(3, inputSize);
    numgrad = computeNumericalGradient(easyJ, easyTheta);
    [~, myGrad] = easyJ(easyTheta);
    fprintf(1, 'Difference between myGrad and numGrad is %f\n', norm(myGrad - numgrad));
end

% First train an autoencoder
theta = initializeParameters(hiddenSize, inputSize);
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
[opttheta, cost] = minFunc(J, theta, options);
                       
% Visualize the learned features
[W1, W2, b1, b2] = unrollParameters(opttheta, inputSize, hiddenSize);
display_network(W1');

% Now train a softmax classifier
trainFeatures = feedForwardAutoencoder(theta, hiddenSize, inputSize, f, trainData);
testFeatures = feedForwardAutoencoder(theta, hiddenSize, inputSize, f, testData);
softmaxModel = softmaxTrain(hiddenSize, numClasses, softmaxLambda, trainFeatures, trainLabels);
% trainFeatures = trainData;
% testFeatures = testData;
% softmaxModel = softmaxTrain(inputSize, numClasses, softmaxLambda, trainFeatures, trainLabels);

% Now compute accuracy on training and test sets
trainPred = softmaxPredict(softmaxModel, trainFeatures);
testPred = softmaxPredict(softmaxModel, testFeatures);

trainCorrect = (trainPred == trainLabels);
testCorrect = (testPred == testLabels);
fprintf(1, 'Training accuracy: %d / %d = %f%%\n', sum(trainCorrect == 1), length(trainCorrect), 100*mean(trainCorrect));
fprintf(1, 'Testing accuracy: %d / %d = %f%%\n', sum(testCorrect == 1), length(testCorrect), 100*mean(testCorrect));




