function [phi, phiPrime] = phiL2(rhoHat, rho)
    phi = (rhoHat - rho).^2;
    phiPrime = 2 * (rhoHat - rho);
end