import unittest
from GradientDescent import GradientDescend as gd
import numpy as np
from Utility import booth_function, get_auto_gradient

class MyTestCase(unittest.TestCase):
    def test_auto_gradient(self):
        f_grad = get_auto_gradient(booth_function)
        grad = f_grad(np.array([1, 1]))
        self.assertAlmostEqual(grad[0], -16, 5)
        self.assertAlmostEqual(grad[1], -20, 5)

    def test_gradient_descent(self):
        x_init = np.zeros(2)
        method = gd(alpha=0.5, beta=0.1, sigma=1e-5)
        x_opt = method.minimize(booth_function, get_auto_gradient(booth_function), x_init)
        method.print_info()
        self.assertAlmostEqual(x_opt[0], 1, 5)
        self.assertAlmostEqual(x_opt[1], 3, 5)

if __name__ == '__main__':
    unittest.main()
