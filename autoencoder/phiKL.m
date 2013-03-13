function [phi, phiPrime] = phiKL(rhoHat, rho)
    phi = rho * log(rho ./ rhoHat) + (1 - rho) * log((1 - rho) ./ (1 - rhoHat));
    phiPrime = ((1 - rho) ./ (1 - rhoHat) - rho ./ rhoHat);
end