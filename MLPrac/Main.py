'''
Created on 2018. 3. 14.

@author: Insup Jung
'''
from MLPrac.GetMin import GetMin #from Package.Filename import Class name
import numpy as np
import matplotlib.pylab as plt


if __name__ == '__main__':
    
    getMin = GetMin() #Have to create instance of Class    
    
    x = np.array([10.0, 10.0])
    x_history = []
    for i in range(100):
        x_history.append(x.copy())   
        x = getMin.get_min(x)    
        
    
    print(x_history)
    
    x_history = np.array(x_history)
    
    plt.plot( [-5, 10], [0,0], '--b')
    plt.plot( [0,0], [-1, 10], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')
    plt.show()
    
    
    pass