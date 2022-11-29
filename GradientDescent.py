from LineSearch import line_search
from scipy.linalg import norm

class GradientDescend:
    def __init__(self, x_init, alpha, beta, sigma):
        self.x_init = x_init
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def minimize(self, f, f_grad):
        x = self.x_init
        gradient = f_grad(x)
        while norm(gradient) < self.sigma:
            delta_x = -gradient
            t = line_search(x, delta_x, f, f_grad, self.alpha, self.beta)
            x = x + t * delta_x
            gradient = f_grad(x)

        return x