from LineSearch import backtracking_line_search
from scipy.linalg import norm

class GradientDescend:
    '''
    x_init: Initial value.
    alpha: Used in backtracking line search, multiple of gradient.
    beta: Used in backtracking line search, multiple of step size.
    sigma: Termination condition on gradient magnitude.
    '''
    def __init__(self, alpha, beta, sigma):
        self.x_init = None
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.info = {
            'iter': 0,
            'grad': [],
            'x': []
        }

    def minimize(self, f, f_grad, x_init):
        self.x_init = x_init.astype('float64')
        x = self.x_init
        gradient = f_grad(x)
        self.info['iter'] = 0
        self.info['grad'] = [norm(gradient)]
        self.info['x'] = [x]

        while norm(gradient) > self.sigma:
            delta_x = -gradient
            t = backtracking_line_search(x, delta_x, f, f_grad, self.alpha, self.beta)
            x = x + t * delta_x
            gradient = f_grad(x)
            self.info['iter'] += 1
            self.info['x'].append(x)
            self.info['grad'].append(norm(gradient))
        return x

    def print_info(self):
        print(self.info)

    def get_info(self):
        return self.info
