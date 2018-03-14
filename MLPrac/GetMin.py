'''
Created on 2018. 3. 14.

@author: Insup Jung
'''

class GetMin:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.alpha = 0.1
        
    def function_1(self, x):
        return x[0]**2 + x[1]**2
    
    def get_differentiate_1(self, x):
        return 2*x[0]
    
    def get_differentiate_2(self, x):
        return 2*x[1]
    
    def get_min(self, x):
        x[0] -= self.alpha*self.get_differentiate_1(x)
        x[1] -= self.alpha*self.get_differentiate_2(x)
        return x

        
        
    
        
    