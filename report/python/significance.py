
import math
from scipy.special import factorial
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson

def SOverSqrtB(s, b):
    return s / np.sqrt(b)

def AsimovSigificance(s,b):
    return np.sqrt( 2 * ( (s+b) * np.log(1 + s/b) - s) )

def poisson(x,lam):
    return np.exp(-lam) * lam**x / factorial(lam)

def PoissonMC(s,b):
    return possion.rvs(s+b, size=1000)

def GaussianMC(s,b):
    mu = s + b
    sigma = np.sqrt(s + b)
    return np.random.normal(mu, sigma, 1000)

if __name__ == "__main__":

    b = np.linspace(0.1,100,10000)

    for s in (2, 5, 10,):
        plt.plot(b, SOverSqrtB(s, b)       , label="s = {}".format(s), color='Red' , linestyle='--', linewidth=0.8)
        plt.plot(b, AsimovSigificance(s, b), label="s = {}".format(s), color='Blue', linestyle='-', linewidth=0.8)

    for s in (2, 5, 10):
        for b in range(0.1, 100, 1):
            x = PoissonMC(s,b)
            x = GaussianMC(s,b)
            p = poisson.cdf(x, b)


    plt.legend()
    plt.xscale("log")
    plt.xlabel("b")
    plt.ylim(0,10)
    plt.ylabel("med[Z_{0}]|s]")
    plt.savefig("discovery_significance.pdf")
