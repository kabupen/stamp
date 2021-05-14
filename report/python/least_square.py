
import numpy as np
import matplotlib.pyplot as plt
import math


model = lambda x : 1/2 * x + 1

xx = np.linspace(0,10,1000)
plt.plot(xx, model(xx))
plt.xlim(0,10)
plt.ylim(0,10)
plt.savefig("LeastSquare.pdf")
