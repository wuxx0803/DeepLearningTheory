function [cost, grad] = sparseLinearAutoencoderCost(theta, visibleSize, hiddenSize, lambda, rho, beta, data)

    n = size(data, 1);
    m = size(data, 2);

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
    
    % Forward propogation for all examples
    a2 = W1 * data + repmat(b1, 1, numExamples);
    a3 = W2 * a2 + repmat(b2, 1, numExamples);
    
    % Compute average activations of all hidden neurons
    rhoHat = mean(a2, 2);
    
    % Backward propogation for all examples
    diff = data - a3;
    delta3 = -diff;
    delta2 = W2' * delta3;
    
    % Square difference term for cost and gradient
    cost = trace(diff * diff') / (2 * numExamples);
    W1grad = (delta2 * data') / numExamples;
    W2grad = (delta3 * a2') / numExamples;
    b1grad = sum(delta2, 2) / numExamples;
    b2grad = sum(delta3, 2) / numExamples;
    
    sparseCost = norm(rhoHat - rho, 2)^2;
    dataSum = sum(data, 2);
    b1gradSparse = (2 / m) * W1 * dataSum + 2 * b1 - 2 * rho;
    
    t1 = (2 / m^2) * W1 * (dataSum * dataSum');
    t2 = (2 / m) * b1 * dataSum';
    t3 = - (2 / m) * rho * ones(hiddenSize, 1) * dataSum';
    W1gradSparse = t1 + t2 + t3;
    
    cost = cost + beta * sparseCost;
    W1grad = W1grad + beta * W1gradSparse;
    b1grad = b1grad + beta * b1gradSparse;
    
    % Add L2 regularization on W1 and W2
    cost = cost + (lambda / 2) * norm(W1, 'fro')^2 + norm(W2, 'fro')^2;
    W1grad = W1grad + lambda * W1;
    W2grad = W2grad + lambda * W2;
    
    % Roll all the gradients up into a long vector
    grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end