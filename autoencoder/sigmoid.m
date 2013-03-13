function [sigm, sigmPrime] = sigmoid(x)  
    ex = exp(-x);
    sigm = 1 ./ (1 + ex);
    sigmPrime = ex ./ (1 + ex).^2;
end