import unittest
import BarrierMethod as BM
import numpy as np
from Utility import get_auto_gradient, get_auto_hessian

class MyTestCase(unittest.TestCase):
    '''
    Create inequality of the form a.T * x - b <= 0
    '''
    def create_affine_inequality(self, a, b):
        def func(x):
            return a@x - b
        return func

    def test_linear_program(self):
        def obj1(x):
            return np.array([-1, 2]) @ x

        def obj2(x):
            return np.array([-1, 0]) @ x

        def obj3(x):
            return np.array([-1, -2]) @ x

        def obj_hess(x):
            n = x.shape[0]
            return np.zeros((n, n))

        fi_1 = self.create_affine_inequality(np.array([2/3, -1]), 2)
        fi_2 = self.create_affine_inequality(np.array([1, 1]), 4)
        fi_3 = self.create_affine_inequality(np.array([-1, 0]), 0)
        fi_ineq = [fi_1, fi_2, fi_3]
        fi_ineq_grad = []
        fi_ineq_hess = []
        for fi in fi_ineq:
            fi_ineq_grad.append(get_auto_gradient(fi))
            fi_ineq_hess.append(get_auto_hessian(fi))
        bm = BM.BarrierMethod(alpha=0.5, beta=0.1, sigma=1e-5, center_sigma=1e-5)

        x_init = np.array([1, 1])

        def run_barrier_method_with_objective(obj):
            x = bm.run_barrier_method(
                obj,
                get_auto_gradient(obj),
                obj_hess,
                fi_ineq,
                fi_ineq_grad,
                fi_ineq_hess,
                x_init,
                A=None,
                b=None)
            return x

        x1 = run_barrier_method_with_objective(obj1)
        x2 = run_barrier_method_with_objective(obj2)
        x3 = run_barrier_method_with_objective(obj3)

        self.assertTrue(np.allclose(x1, np.array([0, -2]), atol=1e-3))
        self.assertTrue(np.allclose(x2, np.array([18/5, 2/5]), atol=1e-3))
        self.assertTrue(np.allclose(x3, np.array([0, 4]), atol=1e-3))

    def test_linear_program_equality_constraint(self):
        def obj1(x):
            return np.array([-1, 2, 0]) @ x

        def obj2(x):
            return np.array([-1, 0, 0]) @ x

        def obj3(x):
            return np.array([-1, -2, 0]) @ x

        def obj_hess(x):
            n = x.shape[0]
            return np.zeros((n, n))

        fi_1 = self.create_affine_inequality(np.array([2 / 3, -1, 0]), 2)
        fi_2 = self.create_affine_inequality(np.array([1, 1, 0]), 4)
        fi_3 = self.create_affine_inequality(np.array([-1, 0, 0]), 0)
        fi_4 = self.create_affine_inequality(np.array([0, 0, 1]), 10)
        fi_ineq = [fi_1, fi_2, fi_3, fi_4]
        fi_ineq_grad = []
        fi_ineq_hess = []
        for fi in fi_ineq:
            fi_ineq_grad.append(get_auto_gradient(fi))
            fi_ineq_hess.append(get_auto_hessian(fi))
        bm = BM.BarrierMethod(alpha=0.5, beta=0.1, sigma=1e-5, center_sigma=1e-5)

        A = np.array([[0, 0, 1]])
        b = np.array([0])

        x_init = np.array([1, 1, 0])

        def run_barrier_method_with_objective(obj):
            x = bm.run_barrier_method(
                obj,
                get_auto_gradient(obj),
                obj_hess,
                fi_ineq,
                fi_ineq_grad,
                fi_ineq_hess,
                x_init,
                A=A,
                b=b)
            return x

        x1 = run_barrier_method_with_objective(obj1)
        x2 = run_barrier_method_with_objective(obj2)
        x3 = run_barrier_method_with_objective(obj3)

        self.assertTrue(np.allclose(x1, np.array([0, -2, 0]), atol=1e-3))
        self.assertTrue(np.allclose(x2, np.array([18 / 5, 2 / 5, 0]), atol=1e-3))
        self.assertTrue(np.allclose(x3, np.array([0, 4, 0]), atol=1e-3))


if __name__ == '__main__':
    unittest.main()
