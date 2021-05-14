# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:51:40 2018

@author: Yuki-F
"""

#Least square method

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    n_data = 11
    
    #Determine function model
    f_model = lambda x: x**2 - 3*x + 2
    x = np.linspace(0, 2, n_data)
    print(x)
    
    #apply noise
    noise = 0.25 * (np.random.rand(1, n_data) - 0.5)
    print(noise)
    y = f_model(x) + noise
    print(y)
    print(type(y))
    
    #Determine function
    p = np.polyfit(x, y.T, 2)
    xx = np.linspace(0, 2, 200)
    yy = np.polyval(p, xx)
    
    #plot data
    plt.plot(x, y.T, 'ro', label = 'Sample') #Sample points
    #plt.hold(True)
    plt.plot(xx, yy, '--', label = 'Determined function') #Interpolation
    #plt.hold(True)
    plt.plot(xx, f_model(xx), label = 'Model function') #model function
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("testp.pdf")
