function [cost,grad] = sparseAutoencoderCostVectorized(theta, visibleSize, hiddenSize, f, phi, ...
                                             lambda, rho, beta, data)
% Parameters:
% theta - weights for the neural net, arranged as a long vector
% visibleSize - Number of input units
% hiddenSize - Number of hidden units
% f - transfer function; must have the form [f(z), f'(z)] = f(z) and
%              operate componentwise on vectors
% phi - Sparsity penalty function; must have the form 
%       [phi(rho_hat, rho), phi'(rho_hat, rho)] = phi(rho_hat, rho)
% lambda - Weight decay parameters
% rho - Desired activation for the hidden units
% beta - Weight of sparsity penalty term
% data - Input data where data(:, i) is the ith training example
  
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

numExamples = size(data, 2);

% Forward propogation for all examples
z2 = W1 * data + repmat(b1, 1, numExamples);
a2 = f(W1 * data + repmat(b1, 1, numExamples));
z3 = W2 * a2 + repmat(b2, 1, numExamples);
a3 = f(z3);

% Compute average activations of all hidden neurons
rhoHat = mean(a2, 2);
% sparseDelta = beta * ((1 - rho) ./ (1 - rhoHat) - rho ./ rhoHat);
[sparseCost, phiPrime] = phi(rhoHat, rho);
sparseDelta = beta * phiPrime;

% Backward propogation for all examples
diff = data - a3;
[~, fp] = f(z3);
delta3 = -diff .* fp;
[~, fp] = f(z2);
delta2 = (W2' * delta3 + repmat(sparseDelta, 1, numExamples)) .* fp;

% Square difference term for cost and gradient
cost = trace(diff * diff') / (2 * numExamples);
W1grad = (delta2 * data') / numExamples;
W2grad = (delta3 * a2') / numExamples;
b1grad = sum(delta2, 2) / numExamples;
b2grad = sum(delta3, 2) / numExamples;

% Add regularization term to cost and gradient
cost = cost + (lambda / 2) * (norm(W1, 'fro')^2 + norm(W2, 'fro')^2);
W1grad = W1grad + lambda * W1;
W2grad = W2grad + lambda * W2;

% Add sparsity term to cost
% sparseCost = sum(rho * log(rho ./ rhoHat) + (1 - rho) * log((1 - rho) ./ (1 - rhoHat)));
cost = cost + beta * sum(sparseCost);

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

