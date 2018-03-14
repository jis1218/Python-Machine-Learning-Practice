#### f(x0, x1) = x0^2 + x1^2의 최솟값을 gradient method로 구하는 방법
```python
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
```

```python
if __name__ == '__main__':
    
    getMin = GetMin() #Have to create instance of Class    
    
    x = np.array([10.0, 10.0]) #넘파이 array를 만들어주고 초기값은 10.0, 10.0으로 한다. 만약 10으로 하면 값이 이상하게 나온다. (int형이라 그런 것 같다.)
    x_history = [] #x_history 배열을 초기화 해준다.
    for i in range(100):
        x_history.append(x.copy()) #x_history에 갱신되는 x값을 추가해준다.
        x = getMin.get_min(x)  #x값은 alpha*편도함수 값으로 구한다.     
    
    print(x_history)
    
    x_history = np.array(x_history)
    
    # 구한 것을 그림으로 나타내는 코드
    plt.plot( [-5, 10], [0,0], '--b')
    plt.plot( [0,0], [-1, 10], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')
    plt.show()   
    pass
```

![img](/img/Figure_1.png)

##### Two layer neural network 구현
```python
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
```

##### 잘 모르는 것
##### > 각 weight의 기울기를 구해주는 코드가 어떻게 나온 것인지 확인이 필요하다.
