import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

X = np.arange(-10,10,0.1)
Y = norm.pdf(X,0,2)

plt.plot(X,Y,color='r')
plt.savefig("gauss.pdf")
