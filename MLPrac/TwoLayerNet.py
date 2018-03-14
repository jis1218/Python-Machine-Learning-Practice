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
        if y.ndim == 1: #ndim은 np.array()로 만든 행렬이 몇차원인지를 알려주는 함수이다.
            t = t.reshape(1, t.size) # t의 형상을 바꿔준다. t가 1차원 배열이라면 ex) [1, 2]이면 형상이 (2, )이지만 reshape(1,2)하게 되면 [[1,2]]로 바뀌게 된다.
            y = y.reshape(1, y.size)
            
        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size: #t.size를 하면 모든 원소의 개수를 구해준다.
            t = t.argmax(axis=1) # 최대값의 인덱스를 반환해준다.
             
        batch_size = y.shape[0] # shape 함수를 쓰게 되면 그 형상을 알 수가 있다.
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
            
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        '''
        Constructor
        '''
        # 가중치 초기화, params라는 변수를 dictionary로 쓰기 위해서는 아래와 같이 params = {} 로 정의해주어야 한다.
        self.params = {};
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size) #np 안에 random 함수에 rand(i, j)를 호출하면 0-1 값을 갖는 iXj의 행렬을 만들어준다.  
        self.params['b1'] = np.zeros(hidden_size) # zeros(i)를 하면 크기가 i이고 값이 모두 0인 행렬을 만들어준다.
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
    
    # x : 입력 데이터, t : 정답 레이블, 실제 데이터 값을 얻어내어 loss function 값을 구한다.
    def loss(self, x, t): 
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predic(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0]) # y와 t의 배열을 index 0부터 탐색하여 같은 index에서 값이 같으면 count를 해준다. ex) y= [[1, 2], [3, 4]] t=[[1, 3], [4, 4]]라면 count는 2 
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) #무슨 코드인지 확인 필요
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 기울기 구해주는 함수
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
        