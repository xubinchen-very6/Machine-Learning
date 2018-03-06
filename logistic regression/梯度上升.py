#梯度上升算法
#fx = -x^2 + 3x - 1

import numpy as np
def diffuc(x_ord):
    return -2 * x_ord + 3

def gradientAscent(x_old = 0,x_new = 1,alpha = 0.01,presision = 0.000001):
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha*diffuc(x_old)
    return np.round(x_new,3)

x_new = gradientAscent()
print(x_new)