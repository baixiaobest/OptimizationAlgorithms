from matplotlib import pyplot as plt
from Utility import get_auto_hessian, get_auto_gradient, booth_function
from NewtonMethod import NewtonMethod as nm
from GradientDescent import GradientDescend as gd
from tests.common import draw_contour
import numpy as np

def run_newton(func, x_init):
    ''' Run newton method '''
    method = nm(alpha=0.5, beta=0.1, sigma=1e-5)
    x_opt = method.minimize(
        func,
        get_auto_gradient(func),
        get_auto_hessian(func),
        x_init)

    info = method.get_info()

    return info

def run_gradient_descent(func, x_init):
    method = gd(alpha=0.5, beta=0.1, sigma=1e-5)
    x_opt = method.minimize(func, get_auto_gradient(func), x_init)

    return method.info

if __name__=="__main__":
    x_init = np.array([3, 0])

    fig, ax = plt.subplots(1, 1)
    draw_contour(booth_function, ax, [-3, 5], [-3, 5])
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

