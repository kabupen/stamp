
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":

    chi_square = lambda chi_2, n : math.e**(-chi_2/2) * chi_2 ** (n/2 - 1 ) / ( 2**(n/2) * math.gamma(n/2) )

    xx = np.linspace(0,28,20000)

    for n in (2,4,10,20,30,40,50):
        plt.plot(xx, chi_square(xx, n), label="n={}".format(n))

    plt.legend(loc="upper right")
    plt.xlim(0,28)
    plt.ylim(0,0.25)

    plt.title("Chi-square distribution")

    plt.savefig("ChiSquareDistribution.pdf")
