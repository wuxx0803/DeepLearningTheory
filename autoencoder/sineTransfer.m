function [f, fPrime] = sineTransfer(x)
    f = sin(x)/2 + 1/2;
    fPrime = cos(x)/2;
end