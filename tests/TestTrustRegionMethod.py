import unittest
from Utility import get_auto_hessian, get_auto_gradient, booth_function, get_banana_function, banana_function
import numpy as np
from TrustRegionMethod import TrustRegionMethod as TR

class MyTestCase(unittest.TestCase):
    def test_trust_region_with_booth(self):
        x_init = np.zeros(2)
        method = TR(sigma=1e-6, trust_region_size=10, max_trust_region_size=100, method='dogleg')
        x_opt = method.minimize(
            booth_function,
            get_auto_gradient(booth_function),
            get_auto_hessian(booth_function),
            x_init)
        method.print_info()
        self.assertAlmostEqual(x_opt[0], 1, 5)
        self.assertAlmostEqual(x_opt[1], 3, 5)

    def test_trust_region_with_rosenbrock(self):
        banana_function = get_banana_function(a=2, b=10)
        x_init = np.array([10, 0])
        method = TR(sigma=1e-6, trust_region_size=10, max_trust_region_size=100, method='dogleg')
        x_opt = method.minimize(
            banana_function,
            get_auto_gradient(banana_function),
            get_auto_hessian(banana_function),
            x_init)
        method.print_info()
        self.assertAlmostEqual(x_opt[0], 2, 5)
        self.assertAlmostEqual(x_opt[1], 4, 5)

if __name__ == '__main__':
    unittest.main()
