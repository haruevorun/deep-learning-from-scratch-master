import numpy as np
x1=3.0
x2=4.0

def function_1(x):
    return x[0]**2 + x[1]**2

def numerical_diff(f,x):
    h = 1e-4#0.0001
    print(h)
    return (f(x+h) - f(x-h)) / (h*2)

#print(numerical_diff(function_1,x1))
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        #print(x[idx])
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad
#print(numerical_gradient(function_1,np.array([3.0,5.0])))

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x

print(gradient_descent(function_1,np.array([-3.0,4.0]), lr=0.1,step_num=100))
