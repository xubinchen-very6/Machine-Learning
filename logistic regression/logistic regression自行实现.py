from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
data = load_iris()
X = data.data[:100,0:2]
x_new = X.tolist()
for i in x_new:
    i.insert(0,1)
X = np.array(x_new)
y = data.target[:100]
class LogisticRegression():
    def __init__(self):
        self.weigh = np.array([[0],[1],[1]]).T
    def sigmoid(self,x):
         return 1/(1+np.exp(-x))
    def gradAscent(self,x, y, alpha, maxCycles):
        for i in range(maxCycles):
            h = self.sigmoid(np.dot(self.weigh,x.T))
            error = y - h  # size:m*1--->1*m
            self.weigh = self.weigh + alpha * np.dot(error,x)
        return self.weigh

lr = LogisticRegression()
weigh = lr.gradAscent(X,y,0.02,10000)
weigh = weigh[0]
def fuc(weigh,x1,x2):
    return 1/(1+np.exp(-(weigh[0]+weigh[1]*x1+weigh[2]*x2)))
print(weigh)
print(fuc(weigh,7,3.2))
plt.scatter(X[:,1],X[:,2],c=y)
plt.show()