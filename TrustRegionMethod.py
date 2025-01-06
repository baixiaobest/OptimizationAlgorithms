import numpy as np
from numpy.linalg import norm, inv, cholesky, LinAlgError, solve
from scipy.linalg import cho_solve

class TrustRegionMethod:
    def __init__(self, sigma=1e-6, trust_region_size=10, max_trust_region_size=100, method='dogleg'):
        '''
        sigma: Threshold of gradient for terminating optimizer.
        trust_region_size: Initial trust regions size.
        '''
        self.sigma = sigma
        self.TR_size = trust_region_size
        self.max_TR_size = max_trust_region_size
        self.reduction_thrhld = 0.25
        self.method = method
        self.info = {
            'iter': 0,
            'grad': [],
            'x': []
        }

    def get_info(self):
        return self.info

    def print_info(self):
        print(self.info)

    def minimize(self, f, f_grad, f_hess, x_init):
        x = x_init
        delta = self.TR_size # Trust region size

        while norm(f_grad(x)) > self.sigma:
            p_k = self._solve_trust_region(f, f_grad, f_hess, x, delta)
            reduction_ratio = (f(x) - f(x + p_k)) / (f(x) - self._quadratic_model(f, f_grad, f_hess, x, p_k))
            # Trust region too big, causing small reduction ratio, need to shrink it.
            if reduction_ratio < 0.25:
                delta = 0.25 * delta
            # Reduction ratio is not too small and still update point is reaching the boundary.
            # Need to expand the trust region.
            elif reduction_ratio > 3/4 and norm(p_k) >= (1-1e-3) * delta:
                delta = min(2*delta, self.max_TR_size)

            # only progress the new iterate if reduction ration is above threshold.
            if reduction_ratio > self.reduction_thrhld:
                x = x + p_k

            self.info['iter'] += 1
            self.info['x'].append(x.copy())
            self.info['grad'].append(f_grad(x))

        return x

    def _solve_trust_region(self, f, f_grad, f_hess, x, delta):
        if self.method == 'dogleg':
            return self._dog_leg_method(f, f_grad, f_hess, x, delta)
        else:
            raise NotImplementedError(f"Method '{self.method}' is not implemented.")

    def _dog_leg_method(self, f, f_grad, f_hess, x, delta):
        g = f_grad(x)
        B_mat = f_hess(x)
        L = np.identity(B_mat.shape[0])
        try:
            L = cholesky(B_mat)
        except LinAlgError:
            print("Hessian needs to be positive definite")
            raise

        p_u = -(g@g/(g@B_mat@g))*g # uncontrained minimizer along steepest descend.
        try:
            p_B = -cho_solve((L, True), g)  # Newton step
        except LinAlgError:
            raise LinAlgError("Failed to solve for Newton step.")

        if delta >= norm(p_B):
            return p_B
        if delta <= norm(p_u):
            return p_u / norm(p_u) * delta
        else:
            p_u2B = p_B - p_u
            coeff_A = p_u2B @ p_u2B
            coeff_B = 2 * p_u @ p_u2B
            coeff_C = p_u @ p_u - delta**2
            alpha = (-coeff_B + np.sqrt(coeff_B**2 - 4 * coeff_A * coeff_C)) / (2*coeff_A)
            return p_u + alpha * p_u2B

    def _quadratic_model(self, f, f_grad, f_hess, x, p):
        return f(x) + f_grad(x) @ p + 0.5 * p @ f_hess(x) @ p
