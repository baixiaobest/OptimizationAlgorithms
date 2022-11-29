from LineSearch import backtracking_line_search
from scipy.linalg import norm

class GradientDescend:
    '''
    x_init: Initial value.
    alpha: Used in backtracking line search, multiple of gradient.
    beta: Used in backtracking line search, multiple of step size.
    sigma: Termination condition on gradient magnitude.
    '''
    def __init__(self, x_init, alpha, beta, sigma):
        self.x_init = x_init
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.info = {
            'iter': 0,
            'grad': []
        }

    def minimize(self, f, f_grad):
        x = self.x_init
        gradient = f_grad(x)
        self.info['iter'] = 0
        self.info['grad'] = []
        while norm(gradient) > self.sigma:
            self.info['grad'].append(norm(gradient))
            delta_x = -gradient
            t = backtracking_line_search(x, delta_x, f, f_grad, self.alpha, self.beta)
            x = x + t * delta_x
            gradient = f_grad(x)
            self.info['iter'] += 1

        return x

    def print_info(self):
        print(self.info)