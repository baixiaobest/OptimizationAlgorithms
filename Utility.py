import numpy as np

'''
Given function f and x,
return gradient of f evaluated at x.
'''
def auto_gradient(f, x, delta=1e-10):
    n = x.shape[0]
    grad = np.zeros(n)
    for i in range(n):
        delta_x = np.zeros(n)
        delta_x[i] = delta
        grad[i] = (f(x + delta_x) - f(x)) / delta

    return grad

'''
Return a gradient function for the given function f.
'''
def get_auto_gradient(f):
    def auto_grad(x):
        return auto_gradient(f, x)

    return auto_grad

def booth_function(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
