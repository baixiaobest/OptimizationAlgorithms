import numpy as np

'''
Given function f and x,
return gradient of f evaluated at x.
'''
def auto_gradient(f, x, delta):
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
def get_auto_gradient(f, delta=1e-10):
    def auto_grad(x):
        return auto_gradient(f, x, delta)

    return auto_grad

def auto_hessian(f, x, delta):
    n = x.shape[0]
    H = np.zeros((n, n))
    for i in range(n):
        delta_x_i = np.zeros(n)
        delta_x_i[i] = delta
        for j in range(i, n):
            delta_x_j = np.zeros(n)
            delta_x_j[j] = delta
            df_dxi_dxj = \
                (f(x + delta_x_i + delta_x_j) + f(x) - f(x + delta_x_i) - f(x + delta_x_j)) \
                / (delta ** 2)
            H[i, j] = df_dxi_dxj
            H[j, i] = df_dxi_dxj

    return H

def get_auto_hessian(f, delta=1e-2):
    def hessian(x):
        return auto_hessian(f, x, delta)
    return hessian

def booth_function(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
