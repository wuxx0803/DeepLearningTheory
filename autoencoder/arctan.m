function [f, fprime] = arctan(x)
    f = atan(x) ./ pi + 1/2;
    fprime = 1 ./ (pi * (1 + x.^2));
end