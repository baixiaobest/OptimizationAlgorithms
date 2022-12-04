import unittest
from NewtonMethod import NewtonMethod as nm
from Utility import get_auto_hessian, get_auto_gradient, booth_function
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_something(self):
        f_hess = get_auto_hessian(booth_function, delta=1e-2)
        h = f_hess(np.array([10, 3]))
        correct_h = np.array([[10, 8], [8, 10]])
        print(h)
        self.assertTrue(np.isclose(h, correct_h, rtol=1e-3, atol=0).all())

    def test_newton(self):
        x_init = np.zeros(2)
        method = nm(x_init, alpha=0.5, beta=0.1, sigma=1e-3)
        x_opt = method.minimize(
            booth_function,
            get_auto_gradient(booth_function),
            get_auto_hessian(booth_function))
        method.print_info()
        self.assertAlmostEqual(x_opt[0], 1, 3)
        self.assertAlmostEqual(x_opt[1], 3, 3)

if __name__ == '__main__':
    unittest.main()
