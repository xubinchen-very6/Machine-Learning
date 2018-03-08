import numpy as np
import matplotlib.pyplot as plt

path = './data.csv'
data = np.loadtxt(path,delimiter=',')
x    = data[:,0].T
y    = data[:,1]
w = np.array([1])
dim = np.shape(x)[0]

class LinearRegression(object):

    def __init__(self,w = np.array([1]),b = 0):
        self.w = w
        self.b = b

    def fit(self,x,y,learning_rate=0.003,epoch=100):
        cost=[]
        for i in range(epoch):
            y_hat = self.w*x+self.b
            error = y - y_hat
            cost.append((1/dim)*np.sum(error**2))
            dw = -(2/dim)*error*x
            dw = np.sum(dw, axis=0)
            db = -(2/dim)*error
            db = np.sum(db, axis=0)
            self.b = self.b - (learning_rate * db)
            self.w = self.w - (learning_rate * dw)
        return self.w,self.b,cost

lr = LinearRegression()
w,b,cost = lr.fit(x,y)
plt.figure(figsize=(10,7))
plt.subplot(121)
plt.scatter(x,y,c='lightblue')
x_show = np.linspace(0,12,100)
plt.title(r'$y=%.3fx+%.3f$'%(w,b))
plt.plot(x_show,w*x_show+b,c='black')
plt.subplot(122)
plt.title('loss fuction')
plt.plot(cost,c='black')
plt.show()






