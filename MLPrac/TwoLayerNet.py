# coding: utf-8
'''
Created on 2018. 3. 14.

@author: Insup Jung
'''
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet(object):
    '''
    classdocs
    '''
    def cross_entropy_error(self, y, t):
        if y.ndim == 1: #ndim�� np.array()�� ���� ����� ������������ �˷��ִ� �Լ��̴�.
            t = t.reshape(1, t.size) # t�� ������ �ٲ��ش�. t�� 1���� �迭�̶�� ex) [1, 2]�̸� ������ (2, )������ reshape(1,2)�ϰ� �Ǹ� [[1,2]]�� �ٲ�� �ȴ�.
            y = y.reshape(1, y.size)
            
        # �Ʒ� �����Ͱ� ��-�� ���Ͷ�� ���� ���̺��� �ε����� ��ȯ
        if t.size == y.size: #t.size�� �ϸ� ��� ������ ������ �����ش�.
            t = t.argmax(axis=1) # �ִ밪�� �ε����� ��ȯ���ش�.
             
        batch_size = y.shape[0] # shape �Լ��� ���� �Ǹ� �� ������ �� ���� �ִ�.
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
            
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        '''
        Constructor
        '''
        # ����ġ �ʱ�ȭ, params��� ������ dictionary�� ���� ���ؼ��� �Ʒ��� ���� params = {} �� �������־�� �Ѵ�.
        self.params = {};
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size) #np �ȿ� random �Լ��� rand(i, j)�� ȣ���ϸ� 0-1 ���� ���� iXj�� ����� ������ش�.  
        self.params['b1'] = np.zeros(hidden_size) # zeros(i)�� �ϸ� ũ�Ⱑ i�̰� ���� ��� 0�� ����� ������ش�.
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predic(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    # x : �Է� ������, t : ���� ���̺�, ���� ������ ���� ���� loss function ���� ���Ѵ�.
    def loss(self, x, t): 
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predic(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0]) # y�� t�� �迭�� index 0���� Ž���Ͽ� ���� index���� ���� ������ count�� ���ش�. ex) y= [[1, 2], [3, 4]] t=[[1, 3], [4, 4]]��� count�� 2 
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) #���� �ڵ����� Ȯ�� �ʿ�
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # ���� �����ִ� �Լ�
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
        