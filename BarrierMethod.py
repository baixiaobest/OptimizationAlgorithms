import numpy as np
from scipy.linalg import norm
import InfeasibleStartNewtonMethod as ISNewton
import NewtonMethod as Newton

'''
Solve problem of the form:
    minimize f(x)
    subject to: A * x = b, 
                fi(x) <= 0 for i from 1 to m.
    A in R^pxn,b in R^p, x in R^n, where p < n 
    f(x) is R^n -> R, convex function
    fi(x) is R^n -> R, convex function
'''
class BarrierMethod:
    def __init__(self, alpha, beta, sigma, center_sigma):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.center_sigma = center_sigma

    '''
        Centering step with linear equality constraint and inequality constraint.
        Problem is transformed into:
            minimize: t*f(x) - sum_i(log(-fi(x)))
            subject to: A*x = b
            
        f(x), fi(x) is R^n -> R
        f_ineq: List of inequality functions.
        f_ineq_grad: List of gradient functions of inequality functions.
        f_ineq_hess: List of hessian matrices of inequality functions.
        x_init needs to be strictly feasible, i.e, A * x_init = b and fi(x_init) < 0.
        A, b: Equality constraint A*x = b
    '''
    def run_barrier_method(self, f, f_grad, f_hess, f_ineq, f_ineq_grad, f_ineq_hess, x_init, A, b):

        m = len(f_ineq)
        n = x_init.shape[0]

        def ff(x, t):
            log_barrier = 0
            for fi in f_ineq:
                neg_fi = -fi(x)
                if neg_fi > 0:
                    log_barrier += -np.log(neg_fi)
                else:
                    log_barrier = np.inf
                    break

            return t*f(x) + log_barrier

        def ff_grad(x, t):
            log_barrier_grad = np.zeros(n)
            for i in range(m):
                fi = f_ineq[i]
                # x is out of domain of -log(-fi(x)).
                if fi(x) >= 0:
                    # Signal the infeasible start newton method
                    # that the dual residual is infinite.
                    return np.ones(n) * np.inf
                fi_grad = f_ineq_grad[i]
                log_barrier_grad += -1/fi(x) * fi_grad(x)

            return t*f_grad(x) + log_barrier_grad

        def ff_hess(x, t):
            log_barrier_hess = np.zeros((n, n))
            for i in range(m):
                fi = f_ineq[i]
                fi_grad = f_ineq_grad[i]
                fi_grad_mat = np.array([fi_grad(x)])
                fi_hess = f_ineq_hess[i]
                log_barrier_hess += 1/(fi(x)**2) * fi_grad_mat.T @ fi_grad_mat - 1/fi(x) * fi_hess(x)

            return t*f_hess(x) + log_barrier_hess

        def get_ff(t):
            def func(x):
                return ff(x, t)
            return func

        def get_ff_grad(t):
            def func(x):
                return ff_grad(x, t)
            return func

        def get_ff_hess(t):
            def func(x):
                return ff_hess(x, t)
            return func

        # TODO: We can initialize t to other more accurate value.
        t = 1
        x = x_init
        while m/t > self.center_sigma:
            if A is None:
                newton = Newton.NewtonMethod(self.alpha, self.beta, self.sigma)
                x = newton.minimize(get_ff(t), get_ff_grad(t), get_ff_hess(t), x)
            else:
                newton = ISNewton.InfeasibleStartNewtonMethod(self.alpha, self.beta, self.sigma)
                x = newton.minimize(get_ff(t), get_ff_grad(t), get_ff_hess(t), x, A, b)
            t = 100*t

        return x
