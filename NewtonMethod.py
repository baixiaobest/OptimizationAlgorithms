from LineSearch import backtracking_line_search
from scipy.linalg import solve

class NewtonMethod:
    def __init__(self, x_init, alpha, beta, sigma):
        self.x_init = x_init
        self.alpha = alpha
        self.beta = beta
        self.sigma_sq = sigma**2
        self.info = {
            'iter': 0,
            'lambda_sq': [],
            'x': []
        }

    def minimize(self, f, f_grad, f_hess):
        x = self.x_init
        grad = f_grad(x)
        H = f_hess(x)
        delta_x = solve(H, -grad)
        lambda_sq_val = -delta_x@grad
        self.info['lambda_sq'] = [lambda_sq_val]
        self.info['x'] = [x]

        while lambda_sq_val > self.sigma_sq:
            grad = f_grad(x)
            delta_x = solve(H, -grad)
            t = backtracking_line_search(x, delta_x, f, f_grad, self.alpha, self.beta)
            x = x + t*delta_x
            delta_x = solve(H, -grad)
            self.info['iter'] += 1
            self.info['lambda_sq'].append(lambda_sq_val)
            self.info['x'].append(x)
            lambda_sq_val = -delta_x@grad

        return x

    def print_info(self):
        print(self.info)

    def get_info(self):
        return self.info
