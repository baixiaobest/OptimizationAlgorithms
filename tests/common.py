from scipy.linalg import norm
import numpy as np

'''
Draw contour of a given function R^2 -> R
'''
def draw_contour(func, ax, x_range, y_range, levels=30):
    ''' Draw the contour of booth function '''
    x_list = np.linspace(x_range[0], x_range[1], 50)
    y_list = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x_list, y_list)
    Z = np.zeros(X.shape)
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            Z[row, col] = func([X[row, col], Y[row, col]])

    ax.contour(X, Y, Z, levels=levels)

'''
Draw a line expressed by a^T x = b, x in R^2
t0: Starting parameter
tf: Ending parameter
res: Parameter interval/resolution.
'''
def draw_line(ax, a, b, t0, tf, res=0.1, color='red'):
    a_norm = norm(a)
    a_hat = np.array(a) / a_norm
    b_hat = np.array(b) / a_norm
    v = np.array([[0, 1], [-1, 0]]) @ a_hat
    T = np.linspace(t0, tf, int((tf-t0) / res))
    X=[]
    Y=[]
    for i in range(T.shape[0]):
        t = T[i]
        pos = a_hat * b_hat + v * t
        X.append(pos[0])
        Y.append(pos[1])

    ax.plot(X, Y, color=color)
