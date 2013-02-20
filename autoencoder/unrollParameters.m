function [W1, W2, b1, b2] = unrollParameters(theta, visibleSize, hiddenSize)
    W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
    b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
end