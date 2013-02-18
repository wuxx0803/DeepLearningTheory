function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

numExamples = size(data, 2);
rho = sparsityParam;

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

z2 = zeros(hiddenSize, numExamples);
a2 = zeros(hiddenSize, numExamples);
z3 = zeros(visibleSize, numExamples);
a3 = zeros(visibleSize, numExamples);

d3 = zeros(visibleSize, numExamples);
d2 = zeros(hiddenSize, numExamples);

for i = 1:numExamples
    % Forward propogation
    z2(:, i) = W1 * data(:, i) + b1;
    a2(:, i) = sigmoid(z2(:, i));
    z3(:, i) = W2 * a2(:, i) + b2;
    a3(:, i) = sigmoid(z3(:, i));
end

meanActivations = mean(a2, 2);

for i = 1:numExamples
    % Back propogation
    d3(:, i) = -(data(:, i) - a3(:, i)) .* a3(:, i) .* (1 - a3(:, i));
    d2(:, i) = W2' * d3(:, i);
    d2(:, i) = d2(:, i) + beta * ( ((1 - rho) ./ (1 - meanActivations)) - (rho ./ meanActivations));
    d2(:, i) = d2(:, i) .* a2(:, i) .* (1 - a2(:, i));
    
    % Update cost
    cost = cost + (0.5 / numExamples) * norm(a3(:, i) - data(:, i))^2;
    
    % Update gradients
    W1grad = W1grad + d2(:, i) * data(:, i)';
    W2grad = W2grad + d3(:, i) * a2(:, i)';
    b1grad = b1grad + d2(:, i);
    b2grad = b2grad + d3(:, i);
end

W1grad = W1grad / numExamples;
W2grad = W2grad / numExamples;
b1grad = b1grad / numExamples;
b2grad = b2grad / numExamples;

% Add regularization term to cost function and gradients
cost = cost + (lambda / 2) * (norm(W1, 'fro')^2 + norm(W2, 'fro')^2);
W1grad = W1grad + lambda * W1;
W2grad = W2grad + lambda * W2;

% Add sparsity term to cost function
sparseCost = sum(rho * log(rho ./ meanActivations) + (1 - rho) * log((1-rho)./(1-meanActivations)));
cost = cost + beta * sparseCost;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
                                     
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

