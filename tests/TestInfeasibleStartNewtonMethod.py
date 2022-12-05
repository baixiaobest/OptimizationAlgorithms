import unittest
from InfeasibleStartNewtonMethod import InfeasibleStartNewtonMethod as ISNM
from scipy.linalg import solve
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_something(self):
        A = np.array([[-1, 1, 3],
                      [3, 1, 1]])
        H = np.array([[2, 1, 0],
                      [1, 5, 6],
                      [0, 6, 1]])
        g = np.array([1, 5, 3])
        h = np.array([3, 9])

        KKT_mat = np.concatenate((H, A.T), axis=1)
        A_aug = np.concatenate((A, np.zeros((2, 2))), axis=1)
        KKT_mat = np.concatenate((KKT_mat, A_aug), axis=0)
        rhs = -np.concatenate((g, h))

        sol = solve(KKT_mat, rhs)
        correct_v = sol[0:3]
        correct_w = sol[3:]
        method = ISNM(alpha=0.5, beta=0.1, sigma=1e-5)
        v, w = method._solve_KKT(H, A, g, h)

        self.assertTrue(np.isclose(v, correct_v, rtol=1e-3, atol=0).all())
        self.assertTrue(np.isclose(w, correct_w, rtol=1e-3, atol=0).all())


if __name__ == '__main__':
    unittest.main()
