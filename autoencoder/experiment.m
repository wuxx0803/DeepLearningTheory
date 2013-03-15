
load('mnist2k/MNIST_train2k.mat');
load('mnist2k/MNIST_test2k.mat');
trainData = X;
trainLabels = y;
testData = Xtest;
testLabels = ytest;
numClasses = max(trainLabels) - min(trainLabels) + 1;

inputSize = size(X, 1);
numTrainingExamples = size(trainData, 2);
numTestExamples = size(testData, 2);

% Draw examples from uniform random distribution
% data = rand(inputSize, numTrainingExamples);

% Draw data from gaussian distribution
% Sigma = [2, 1; 1, 3]; R = chol(Sigma);
% R = [sqrt(2), sqrt(2); -sqrt(2), sqrt(2)] * diag([1, 5]);
% data = randn(numTrainingExamples, 2) * R;
% data = data';

checkGradient = false;
hiddenSize = 200;
% lambda = 3e-3;
lambda = 0;
beta = 10;
rho = 0.05;
f = @sineTransfer;
phi = @phiNone;
softmaxLambda = 1e-4;

% Lets do PCA
% Mean subtraction:
% b = mean(trainData, 2);
% Xc = X - repmat(b, 1, numTrainingExamples);
% [U, S, V] = svd(Xc, 0);
% U = U(:, 1:hiddenSize);
% display_network(U);
% pause;

% Define the objective function
J = @(p) generalSparseAutoencoderCost(p, inputSize, hiddenSize, f, phi, lambda, rho, beta, trainData);

% Do numeric gradient checking
if checkGradient
    easyJ = @(p) generalSparseAutoencoderCost(p, inputSize, 3, f, phi, lambda, rho, beta, trainData(:, 1:10));
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
trainFeatures = generalFeedForwardAutoencoder(opttheta, hiddenSize, inputSize, f, trainData);
testFeatures = generalFeedForwardAutoencoder(opttheta, hiddenSize, inputSize, f, testData);
% trainFeatures = U' * (trainData - repmat(b, 1, numTrainingExamples));
% testFeatures = U' * (testData - repmat(b, 1, numTestExamples));
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




