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

    def minimize(self, f, f_grad, f_hess, f_ineq_list, f_ineq_grad_list, f_ineq_hess_list, x_init, A, b):
        x, s, _, _ = self.solve_feasibility(f_ineq_list, f_ineq_grad_list, f_ineq_hess_list, x_init, A, b)
        if s > 0:
            print("Problem is infeasible")
            return None, None, None
        return self.run_barrier_method(f, f_grad, f_hess, f_ineq_list, f_ineq_grad_list, f_ineq_hess_list, x, A, b)

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
        Original problem is transformed into:
            minimize: t*f(x) - Σ_i( log(-fi(x)) )
            subject to: A*x = b
            
        and it is solved with infeasible start newton method 
        or newton method in there is no equality constraint.   
        t is scaling constant
        f(x), fi(x) is R^n -> R
        
        f: Original objective function f(x).
        f_grad: Gradient of original objective function.
        f_hess: Hessian of original objective function.
        f_ineq_list: List of inequality functions.
        f_ineq_grad_list: List of gradient functions of inequality functions.
        f_ineq_hess_list: List of hessian matrices of inequality functions.
        x_init needs to be strictly feasible, i.e, A * x_init = b and fi(x_init) < 0.
        A, b: Equality constraint A*x = b
        term_pos_dual: Terminate the iteration when iteration enters positive dual objective.
            Dual objective is f(x) - m/t
    '''
    def run_barrier_method(self, f, f_grad, f_hess, f_ineq_list, f_ineq_grad_list, f_ineq_hess_list,
                           x_init, A, b, term_pos_dual=False):

        m = len(f_ineq_list)
        x_init = x_init.astype('float64')

        for fi in f_ineq_list:
            if fi(x_init) > 0:
                raise ValueError("x_init needs to be feasible in inequality constraints.")

        def get_log_barrier_grad(f_ineq_list, f_ineq_grad_list):
            def func(x):
                return self._log_barrier_grad(x, f_ineq_list, f_ineq_grad_list)
            return func

        t = self.calculate_starting_t(
            x_init, f_grad, get_log_barrier_grad(f_ineq_list, f_ineq_grad_list), A)
        duality_gap = m / t

        x = x_init
        f_center = 0 # Objective value after each centering step.
        while duality_gap > self.center_sigma and (not term_pos_dual or f_center - duality_gap <= 0):
            if A is None:
                newton = Newton.NewtonMethod(self.alpha, self.beta, self.sigma)
                x = newton.minimize(
                    self._get_ff(f, f_ineq_list, t),
                    self._get_ff_grad(f_grad, f_ineq_list, f_ineq_grad_list, t),
                    self._get_ff_hess(f_hess, f_ineq_list, f_ineq_grad_list, f_ineq_hess_list, t),
                    x)
            else:
                newton = ISNewton.InfeasibleStartNewtonMethod(self.alpha, self.beta, self.sigma)
                x, v = newton.minimize(
                    self._get_ff(f, f_ineq_list, t),
                    self._get_ff_grad(f_grad, f_ineq_list, f_ineq_grad_list, t),
                    self._get_ff_hess(f_hess, f_ineq_list, f_ineq_grad_list, f_ineq_hess_list, t),
                    x, A, b)

            f_center = f(x)
            duality_gap = m / t
            t = 100*t

        if term_pos_dual and f_center - duality_gap > 0:
            print(f"terminate with dual objective value: {f_center - duality_gap}")

        # Calculate dual variable for inequalities.
        λ = np.zeros(m)
        for i in range(m):
            fi = f_ineq_list[i]
            λ[i] = -1 / (t * fi(x))

        if A is None:
            return x, λ, None
        else:
            return x, λ, v/t

    '''
    This function solve the feasibility problem, systems of inequalities and equalities :
        fi(x) <= 0 and A@x = b
        
    The above problem is formulated as:
        minimize s
        subject to: fi(x) <= s
                    A@x = b
    If the solution of the above problem has primal objective less than equal to 0, i.e. s <= 0,
    then we have a solution to the systems of inequalities and equalities.
    
    The above problem again is transformed into objective with log barrier:
        minimize s - Σ_i( log(s - fi(x)) )
        subject to: A@x = b
    and solved with infeasible start newton method.
    Variables are x in R_n and s in R.
    '''
    def solve_feasibility(self, f_ineq_list, f_ineq_grad_list, f_ineq_hess_list, x_init, A, b):
        m = len(f_ineq_list)
        n = x_init.shape[0]
        x_init = x_init.astype('float64')

        # Here, var is [x, s], x in R_n, s in R
        def f_mod(var):
            return var[-1]

        def f_grad_mod(var):
            l = var.shape[0]
            grad = np.zeros(l)
            grad[-1] = 1
            return grad

        def f_hess_mod(var):
            l = var.shape[0]
            hess = np.zeros((l, l))
            return hess

        # fi is R^n -> R, inequality function.
        # modified inequality fi is fi_mod(x, s) = fi(x) - s
        def get_fi_ineq_mod(fi):
            def fi_ineq_mod(var):
                x = var[0:-1]
                s = var[-1]
                return fi(x) - s

            return fi_ineq_mod

        # fi_grad is R^n -> R^n, gradient function.
        # Modified gradient of fi is ∇fi_mod(x, s) = [∇fi(x), -1]
        def get_fi_ineq_grad_mod(fi_grad):
            def fi_ineq_grad_mod(var):
                x = var[0:-1]
                grad = np.concatenate((fi_grad(x), np.array([-1])))
                return grad
            return fi_ineq_grad_mod

        # Modified hessian of inequality fi is ∇^2 fi_mod(x, s) = [[∇^2fi(x), 0],
        #                                                          [0,         0]]
        def get_fi_ineq_hess_mod(fi_hess):
            def fi_ineq_hess_mod(var):
                l = var.shape[0]
                n = l-1
                x = var[0:-1]
                hess = np.zeros((l, l))
                hess[0:n, 0:n] = fi_hess(x)

                return hess
            return fi_ineq_hess_mod

        f_ineq_mods = []
        f_ineq_grad_mods = []
        f_ineq_hess_mods = []
        s_init = 0.0
        for i in range(m):
            fi = f_ineq_list[i]
            fi_grad = f_ineq_grad_list[i]
            fi_hess = f_ineq_hess_list[i]
            f_ineq_mods.append(get_fi_ineq_mod(fi))
            f_ineq_grad_mods.append(get_fi_ineq_grad_mod(fi_grad))
            f_ineq_hess_mods.append(get_fi_ineq_hess_mod(fi_hess))

            s_init = max(s_init, fi(x_init))

        s_init = s_init + 1
        var_init = np.concatenate((x_init, np.array([s_init])))
        A_aug = None
        if not A is None:
            A_aug = np.concatenate((A, np.zeros((m, 1))), axis=1)

        var, λ, v = \
            self.run_barrier_method(
                f_mod, f_grad_mod, f_hess_mod,
                f_ineq_mods, f_ineq_grad_mods, f_ineq_hess_mods,
                var_init, A_aug, b, term_pos_dual=True)

        return var[0:n], var[-1], λ, v

    def _get_ff(self, f, f_ineq_list, t):
        # Function t*f(x) - sum_i(log(-fi(x)))
        def ff(x):
            log_barrier = 0
            for fi in f_ineq_list:
                neg_fi = -fi(x)
                if neg_fi > 0:
                    log_barrier += -np.log(neg_fi)
                else:
                    log_barrier = np.inf
                    break

            return t * f(x) + log_barrier

        return ff

    # gradient of log barrier, d/dx (- sum_i(log(-fi(x))))
    def _log_barrier_grad(self, x, f_ineq_list, f_ineq_grad_list):
        n = x.shape[0]
        m = len(f_ineq_list)
        lb_grad = np.zeros(n)
        for i in range(m):
            fi = f_ineq_list[i]
            # x is out of domain of -log(-fi(x)).
            if fi(x) >= 0:
                # Signal the infeasible start newton method
                # that the dual residual is infinite.
                return np.ones(n) * np.inf
            fi_grad = f_ineq_grad_list[i]
            lb_grad += -1 / fi(x) * fi_grad(x)

        return lb_grad

    def _get_ff_grad(self, f_grad, f_ineq_list, f_ineq_grad_list, t):
        # Gradient of objective: d/dx (t*f(x) - sum_i(log(-fi(x))))
        def ff_grad(x):
            lb_grad = self._log_barrier_grad(x, f_ineq_list, f_ineq_grad_list)

            return t * f_grad(x) + lb_grad

        return ff_grad

    def _get_ff_hess(self, f_hess, f_ineq_list, f_ineq_grad_list, f_ineq_hess_list, t):
        # Hessian of objective: d^2/dx^2 (t*f(x) - sum_i(log(-fi(x))))
        def ff_hess(x):
            n = x.shape[0]
            m = len(f_ineq_list)
            log_barrier_hess = np.zeros((n, n))
            for i in range(m):
                fi = f_ineq_list[i]
                fi_grad = f_ineq_grad_list[i]
                fi_grad_mat = np.array([fi_grad(x)])
                fi_hess = f_ineq_hess_list[i]
                log_barrier_hess += 1 / (fi(x) ** 2) * fi_grad_mat.T @ fi_grad_mat - 1 / fi(x) * fi_hess(x)

            return t * f_hess(x) + log_barrier_hess

        return ff_hess
