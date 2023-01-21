import numpy as np
import InfeasibleStartNewtonMethod as ISNewton
import NewtonMethod as Newton
from scipy.linalg import lstsq

'''
Solve problem of the form:
    minimize f(x)
    subject to: A * x = b, 
                fi(x) <= 0 for i from 1 to m.
    A in R^pxn,b in R^p, x in R^n, where p < n 
    f(x) is R^n -> R, convex function
    fi(x) is R^n -> R, convex function
    
This problem will be transformed and solved into this form:
    minimize: t*f(x) - sum_i(log(-fi(x)))
    subject to: A*x = b
    f(x), fi(x) is R^n -> R
'''
class BarrierMethod:
    def __init__(self, alpha, beta, sigma, center_sigma):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.center_sigma = center_sigma

    '''
    t value can be chosen such that it minimizes:
        inf_v || t * ∇f(x_init) + ∇log_barrier(x_init) + A.T@v ||2
    This can be found by solving least square of the form:
        minimize || [∇f(x_init), A.T] @ [[t], [v]] + ∇log_barrier(x_init) ||2 
    Or in normal equation:
        M.T @ M @ [[t], [v]] = -M.T@∇log_barrier(x_init), 
        M = [∇f(x_init), A.T]
    '''
    def calculate_starting_t(self, x_init, f_grad, log_barrier_grad, A):
        f_grad_init = f_grad(x_init)
        lb_grad_init = log_barrier_grad(x_init)
        M = None
        if A is None:
            M = np.array([f_grad_init]).T
        else:
            M = np.concatenate((np.array([f_grad_init]).T, A.T), axis=1)
        x_hat, res, r, s = lstsq(M, -np.array([lb_grad_init]).T)

        return np.abs(x_hat[0, 0])

    '''
        Centering step with linear equality constraint and inequality constraint.
        Problem is transformed into:
            minimize: t*f(x) - sum_i(log(-fi(x)))
            subject to: A*x = b
            
        f(x), fi(x) is R^n -> R
        f: Original objective function f(x).
        f_grad: Gradient of original objective function.
        f_hess: Hessian of original objective function.
        f_ineq: List of inequality functions.
        f_ineq_grad: List of gradient functions of inequality functions.
        f_ineq_hess: List of hessian matrices of inequality functions.
        x_init needs to be strictly feasible, i.e, A * x_init = b and fi(x_init) < 0.
        A, b: Equality constraint A*x = b
        term_pos_dual: Terminate the iteration when iteration enters positive dual objective.
            Dual objective is f(x) - m/t
    '''
    def run_barrier_method(self, f, f_grad, f_hess, f_ineq, f_ineq_grad, f_ineq_hess, x_init, A, b, term_pos_dual=False):

        m = len(f_ineq)

        def get_log_barrier_grad(f_ineq, f_ineq_grad):
            def func(x):
                return self._log_barrier_grad(x, f_ineq, f_ineq_grad)
            return func

        t = self.calculate_starting_t(
            x_init, f_grad, get_log_barrier_grad(f_ineq, f_ineq_grad), A)

        x = x_init
        f_center = 0 # Objective value after each centering step.
        while m/t > self.center_sigma and (not term_pos_dual or f_center - m / t <= 0):
            if A is None:
                newton = Newton.NewtonMethod(self.alpha, self.beta, self.sigma)
                x = newton.minimize(
                    self._get_ff(f, f_ineq, t),
                    self._get_ff_grad(f_grad, f_ineq, f_ineq_grad, t),
                    self._get_ff_hess(f_hess, f_ineq, f_ineq_grad, f_ineq_hess, t),
                    x)
            else:
                newton = ISNewton.InfeasibleStartNewtonMethod(self.alpha, self.beta, self.sigma)
                x, v = newton.minimize(
                    self._get_ff(f, f_ineq, t),
                    self._get_ff_grad(f_grad, f_ineq, f_ineq_grad, t),
                    self._get_ff_hess(f_hess, f_ineq, f_ineq_grad, f_ineq_hess, t),
                    x, A, b)

            f_center = f(x)
            t = 100*t

        # Calculate dual variable for inequalities.
        λ = np.zeros(m)
        for i in range(m):
            fi = f_ineq[i]
            λ[i] = -1 / (t * fi(x))

        if A is None:
            return x, λ, None
        else:
            return x, λ, v/t

    def run_phase_1_feasibility(self):
        pass

    def _get_ff(self, f, f_ineq, t):
        # Function t*f(x) - sum_i(log(-fi(x)))
        def ff(x):
            log_barrier = 0
            for fi in f_ineq:
                neg_fi = -fi(x)
                if neg_fi > 0:
                    log_barrier += -np.log(neg_fi)
                else:
                    log_barrier = np.inf
                    break

            return t * f(x) + log_barrier

        return ff

    # gradient of log barrier, d/dx (- sum_i(log(-fi(x))))
    def _log_barrier_grad(self, x, f_ineq, f_ineq_grad):
        n = x.shape[0]
        m = len(f_ineq)
        lb_grad = np.zeros(n)
        for i in range(m):
            fi = f_ineq[i]
            # x is out of domain of -log(-fi(x)).
            if fi(x) >= 0:
                # Signal the infeasible start newton method
                # that the dual residual is infinite.
                return np.ones(n) * np.inf
            fi_grad = f_ineq_grad[i]
            lb_grad += -1 / fi(x) * fi_grad(x)

        return lb_grad

    def _get_ff_grad(self, f_grad, f_ineq, f_ineq_grad, t):
        # Gradient of objective: d/dx (t*f(x) - sum_i(log(-fi(x))))
        def ff_grad(x):
            lb_grad = self._log_barrier_grad(x, f_ineq, f_ineq_grad)

            return t * f_grad(x) + lb_grad

        return ff_grad

    def _get_ff_hess(self, f_hess, f_ineq, f_ineq_grad, f_ineq_hess, t):
        # Hessian of objective: d^2/dx^2 (t*f(x) - sum_i(log(-fi(x))))
        def ff_hess(x):
            n = x.shape[0]
            m = len(f_ineq)
            log_barrier_hess = np.zeros((n, n))
            for i in range(m):
                fi = f_ineq[i]
                fi_grad = f_ineq_grad[i]
                fi_grad_mat = np.array([fi_grad(x)])
                fi_hess = f_ineq_hess[i]
                log_barrier_hess += 1 / (fi(x) ** 2) * fi_grad_mat.T @ fi_grad_mat - 1 / fi(x) * fi_hess(x)

            return t * f_hess(x) + log_barrier_hess

        return ff_hess
