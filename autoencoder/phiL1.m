function [phi, phiPrime] = phiL1(rhoHat, rho)
    phi = abs(rhoHat - rho);
    phiPrime = sign(rhoHat - rho);
end