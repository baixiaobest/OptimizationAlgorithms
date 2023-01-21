from Utility import get_auto_hessian, get_auto_gradient, booth_function, get_banana_function
from InfeasibleStartNewtonMethod import InfeasibleStartNewtonMethod as ISNM
from common import draw_contour, draw_line
import numpy as np
from matplotlib import pyplot as plt

if __name__=="__main__":
    test_setup_group = {
        'booth': {
            'func': booth_function,
            'x_range': [-10, 10],
            'y_range': [-10, 10],
            'x_init': np.array([-1, 0])
        },
        'banana': {
            'func': get_banana_function(a=1, b=5),
            'x_range': [-2, 2],
            'y_range': [-2, 2],
            'x_init': np.array([0, -1])
        }
    }
    test_setup = test_setup_group['booth']

    x_init = test_setup['x_init']
    fig, ax = plt.subplots(1, 1)
    draw_contour(test_setup['func'], ax, test_setup['x_range'], test_setup['y_range'], levels=100)

    a = np.array([1, 1])
    b = 0
    draw_line(ax, a, b, -10, 10, res=0.1)

    A = np.array([a])
    B = np.array([b])
    method = ISNM(alpha=0.5, beta=0.1, sigma=1e-5)
    x_opt, v = method.minimize(
        booth_function,
        get_auto_gradient(booth_function),
        get_auto_hessian(booth_function),
        x_init,
        A,
        B)

    info = method.get_info()
    ax.plot([x[0] for x in info['x']], [x[1] for x in info['x']], marker='x')

    fig2, axs = plt.subplots(1, 1)
    axs.plot([*range(len(info['residual_norm']))], info['residual_norm'], marker='x')
    axs.set_yscale('log')

    plt.show()