
import math
from scipy.special import factorial
import numpy as np
import matplotlib.pyplot as plt

def poisson(x, lam):
    return math.exp(-lam) * lam**x / factorial(x)

if __name__ == "__main__":

    x = np.linspace(0,40,1000)

    for lam in (1,2,3,4,5,10,20,30):
        plt.plot(x, poisson(x,lam), label="lambda={}".format(lam))

    plt.legend()
    plt.savefig("poisson_scan.pdf")
