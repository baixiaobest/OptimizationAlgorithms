def line_search(x, delta_x, f, f_grad, alpha, beta):
    f_x = f(x)
    f_grad_x = f_grad(x)
    t = 1.0
    while f(x + t*delta_x) > f_x + alpha * t * f_grad_x @ delta_x:
        t *= beta
    return t