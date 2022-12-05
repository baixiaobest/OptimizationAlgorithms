from Utility import get_auto_hessian, get_auto_gradient, booth_function
from InfeasibleStartNewtonMethod import InfeasibleStartNewtonMethod as ISNM
from common import draw_contour, draw_line
import numpy as np
from matplotlib import pyplot as plt

if __name__=="__main__":
    x_init = np.array([-1, 0])
    fig, ax = plt.subplots(1, 1)
    draw_contour(booth_function, ax, [-10, 10], [-10, 10], levels=100)

    a = np.array([1, 1])
    b = -2
    draw_line(ax, a, b, -10, 10, res=0.1)

    A = np.array([a])
    B = np.array([b])
    method = ISNM(alpha=0.5, beta=0.1, sigma=1e-5)
    x_opt = method.minimize(
        booth_function,
        get_auto_gradient(booth_function),
        get_auto_hessian(booth_function),
        x_init,
        A,
        B)

    info = method.get_info()
    ax.plot([x[0] for x in info['x']], [x[1] for x in info['x']], marker='x')
    plt.show()