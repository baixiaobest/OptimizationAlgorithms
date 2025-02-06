import numpy as np

def backtracking_line_search(x, delta_x, f, f_grad, alpha, beta):
    f_x = f(x)
    f_grad_x = f_grad(x)
    t = 1.0
    while f(x + t*delta_x) > f_x + alpha * t * f_grad_x @ delta_x:
        t *= beta
    return t

def strong_wolfe_line_search(x, delta_x, f, f_grad, a_max, c1, c2, a_growth=0.1, iteration=50):
    '''
    Perform line search on function f(x) and find a that satisfies the following strong Wolfe's conditions:
        1. f(x + a * delta_x) <= f(x) + c1 * a * f_grad(x)^T * delta_x
        2. |f_grad(x + a * delta_x)^T * delta_x| <= c2 * |f_grad(x)^T * delta_x|

        0 < c1 < c2 < 1.
        delta_x is unit length.
    '''
    def phi(a):
        return f(x+a*delta_x)

    def phi_grad(a):
        return f_grad(x + a*delta_x) @ delta_x

    first_guess_factor = 0.5

    phi_0 = phi(0)
    phi_grad_0 = phi_grad(0)
    a_i = a_max * first_guess_factor
    a_i_prev = 0

    for i in range(iteration):
        if a_i > a_max:
            return False, 0

        phi_ai = phi(a_i)
        if phi_ai > phi_0 + c1 * a_i * phi_grad_0 \
            or (i >= 1 and phi_ai >= phi(a_i_prev)):
            return zoom(phi, phi_grad, a_i_prev, a_i, c1, c2)

        phi_grad_ai = phi_grad(a_i)
        if np.abs(phi_grad_ai) <= -c2 * phi_grad_0:
            return True, a_i

        if phi_grad_ai > 0:
            return zoom(phi, phi_grad, a_i, a_i_prev, c1, c2)

        a_i_prev = a_i
        a_i = (1 + a_growth) * a_i

    return False, 0


def zoom(phi, phi_grad, a_lo, a_hi, c1, c2, iteration=50):
    phi_0 = phi(0)
    phi_grad_0 = phi_grad(0)

    for i in range(iteration):
        phi_a_lo = phi(a_lo)
        a_mid = (a_lo + a_hi) / 2
        phi_a_mid = phi(a_mid)

        if phi_a_mid > phi_0 + c1 * a_mid * phi_grad_0 or phi_a_mid >= phi_a_lo:
            a_hi = a_mid
        else:
            phi_grad_a_mid = phi_grad(a_mid)
            if np.abs(phi_grad_a_mid) <= -c2 * phi_grad_0:
                return True, a_mid
            if phi_grad_a_mid * (a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a_mid

    return False, 0
