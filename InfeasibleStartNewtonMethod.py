from scipy.linalg import solve, solve_triangular, norm, cholesky
import numpy as np

class InfeasibleStartNewtonMethod:
    '''
    alpha: Line search alpha.
    beta: Line search step multiplier beta.
    sigma: Tolerance for norm of primal residual and tolerance for norm of residuals.
        Use for program termination.
    max_iter: Maximum iteration.
    '''
    def __init__(self, alpha, beta, sigma, max_iter=100):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.max_iter = max_iter
        self.x_init = None
        self.A = None
        self.b = None
        self.f_grad = None
        self.info = {
            'iter': 0,
            'residual_norm': [],
            'x': []
        }

    '''
    Minimize function with linear equality constraint.
    minimize f(x)
    subject to: Ax = b
    A in R^pxn,b in R^p, x in R^n, where p < n 
    f(x) is R^n -> R
    x_init needs to be in domain of f but no need to be feasible.
    '''
    def minimize(self, f, f_grad, f_hess, x_init, A, b):
        x_init = x_init.astype('float64')
        A = A.astype('float64')
        b = b.astype('float64')
        self.x_init = x_init
        self.A = A
        self.b = b
        self.f_grad = f_grad

        x = x_init
        v = np.zeros(A.shape[0])
        r_dual, r_pri = self._residual(x, v)
        r = np.concatenate((r_dual, r_pri), axis=0)

        self.info['residual_norm'].append(norm(r))
        self.info['x'].append(x)

        num_iter = 0
        while (r_pri > self.sigma or norm(r) > self.sigma) and num_iter < self.max_iter:
            H = f_hess(x)
            x_delta, v_delta = self._solve_KKT(H, A, r_dual, r_pri)
            t = self._line_search(x, v, x_delta, v_delta)
            x = x + t * x_delta
            v = v + t * v_delta
            r_dual, r_pri = self._residual(x, v)
            r = np.concatenate((r_dual, r_pri), axis=0)

            self.info['iter'] += 1
            self.info['residual_norm'].append(norm(r))
            self.info['x'].append(x)

            num_iter += 1

        return x

    def get_info(self):
        return self.info

    def _line_search(self, x, v, x_delta, v_delta):
        alpha = self.alpha
        beta = self.beta
        t = 1.0
        r_norm_init = self._residual_norm(x, v)
        while self._residual_norm(x + t*x_delta, v + t*v_delta) > (1 - alpha*t) * r_norm_init:
            t *= beta

        return t

    def _residual_norm(self, x, v):
        r_dual, r_pri = self._residual(x, v)
        vec = np.concatenate((r_dual, r_pri), axis=0)
        return norm(vec)

    def _residual(self, x, v):
        A = self.A
        b = self.b
        f_grad = self.f_grad
        r_dual = f_grad(x) + A.T@v
        r_pri = A@x - b
        return r_dual, r_pri

    '''
    Solve the KKT matrix of the form
    [[H, A.T],    [[v],         [[g],
     [A, 0]]   *   [w]]    =  -  [h]]
     assuming H is symmetric. 
     Note right hand side has a negative sign.
    '''
    def _solve_KKT(self, H, A, g, h):
        Ag = np.concatenate((A.T, np.array([g]).T), axis=1)
        L = cholesky(H, lower=True)
        H_inv_Ag = self._solve_cholesky(L, Ag)
        H_inv_A = H_inv_Ag[:, 0:-1]
        H_inv_g = H_inv_Ag[:, -1]

        S = -A@H_inv_A
        w = solve(S, A@H_inv_g - h, assume_a='sym')
        v = self._solve_cholesky(L, -A.T @ w - g)

        return v, w

    '''
    Solve the equation of the form: LL^T x = b
    '''
    def _solve_cholesky(self, L, b):
        x = solve_triangular(L, b, lower=True)
        y = solve_triangular(L.T, x, lower=False)
        return y
