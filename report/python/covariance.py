
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__" :

    for cov in range(-1,2):
        mean = np.array([0, 0]) # 平均を指定。
    
        cov_matrix = np.array([
            [1,  cov],
            [cov, 1]]) 
    
        x, y = np.random.multivariate_normal(mean, cov_matrix, 5000).T 
        
        plt.figure()
        plt.plot(x, y, '.') 
    
        plt.title("covariance = ".format(cov))
        plt.axis("equal")
        plt.savefig('covariance_{}.png'.format(cov))
