from matplotlib import pyplot as plt
from Utility import get_auto_hessian, get_auto_gradient, booth_function
from NewtonMethod import NewtonMethod as nm
from GradientDescent import GradientDescend as gd
import numpy as np

def draw_function(func):
    ''' Draw the contour of booth function '''
    x_list = np.linspace(-3, 5, 50)
    y_list = np.linspace(-3, 5, 50)
    X, Y = np.meshgrid(x_list, y_list)
    Z = np.zeros(X.shape)
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            Z[row, col] = func([X[row, col], Y[row, col]])

    fig, ax = plt.subplots(1, 1)
    ax.contour(X, Y, Z, levels=30)

    return fig, ax

def run_newton(func, x_init):
    ''' Run newton method '''
    method = nm(x_init, alpha=0.5, beta=0.1, sigma=1e-5)
    x_opt = method.minimize(
        booth_function,
        get_auto_gradient(func),
        get_auto_hessian(func))

    info = method.get_info()

    return info

def run_gradient_descent(func, x_init):
    method = gd(x_init, alpha=0.5, beta=0.1, sigma=1e-5)
    x_opt = method.minimize(booth_function, get_auto_gradient(func))

    return method.info

if __name__=="__main__":
    x_init = np.array([3, 0])
    fig, ax = draw_function(booth_function)
    newton_info = run_newton(booth_function, x_init)
    gradient_info = run_gradient_descent(booth_function, x_init)

    ax.plot([x[0] for x in newton_info['x']], [x[1] for x in newton_info['x']], marker='x')
    ax.plot([x[0] for x in gradient_info['x']],
            [x[1] for x in gradient_info['x']],
            marker='x',
            color='red')

    fig2, axs = plt.subplots(2, 1)
    axs[0].plot(range(len(newton_info['lambda_sq'])), newton_info['lambda_sq'], marker='x')
    axs[0].set_yscale('log')
    axs[1].plot(range(len(gradient_info['grad'])), gradient_info['grad'], marker='x')
    axs[1].set_yscale('log')

    plt.show()

