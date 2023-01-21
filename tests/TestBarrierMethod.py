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

        x1, λ1, v1 = run_barrier_method_with_objective(obj1)
        x2, λ2, v2 = run_barrier_method_with_objective(obj2)
        x3, λ2, v2 = run_barrier_method_with_objective(obj3)

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

        x1, λ1, v1 = run_barrier_method_with_objective(obj1)
        x2, λ2, v2 = run_barrier_method_with_objective(obj2)
        x3, λ2, v2 = run_barrier_method_with_objective(obj3)

        self.assertTrue(np.allclose(x1, np.array([0, -2, 0]), atol=1e-3))
        self.assertTrue(np.allclose(x2, np.array([18 / 5, 2 / 5, 0]), atol=1e-3))
        self.assertTrue(np.allclose(x3, np.array([0, 4, 0]), atol=1e-3))

    def test_feasibility(self):
        fi_1 = self.create_affine_inequality(np.array([2 / 3, -1]), 2)
        fi_2 = self.create_affine_inequality(np.array([1, 1]), 4)
        fi_3 = self.create_affine_inequality(np.array([-1, 0]), 0)
        fi_ineq = [fi_1, fi_2, fi_3]
        fi_ineq_grad = []
        fi_ineq_hess = []
        for fi in fi_ineq:
            fi_ineq_grad.append(get_auto_gradient(fi))
            fi_ineq_hess.append(get_auto_hessian(fi))

        bm = BM.BarrierMethod(alpha=0.5, beta=0.1, sigma=1e-5, center_sigma=1e-5)

        x_init = np.array([100, 100])
        x, s, _, _ = bm.solve_feasibility(fi_ineq, fi_ineq_grad, fi_ineq_hess, x_init, None, None)

        self.assertTrue(s <= 0)
        for fi in fi_ineq:
            self.assertTrue(fi(x) <= 0)

    def test_feasibility_2(self):
        np.random.seed(4)
        f_ineq = []
        x_sol = np.array([10, 21, 32, -12, 34, 23, 53, 12, 42, 100])
        tolerance = 1
        n = 10
        N = 100
        for i in range(N):
            a_rand = np.random.rand(n) * 10
            b = a_rand@x_sol + tolerance
            f_ineq.append(self.create_affine_inequality(a_rand, b))

        fi_ineq_grad = []
        fi_ineq_hess = []
        s_max_init = 0
        x_init = np.ones(n) * 1000
        for fi in f_ineq:
            fi_ineq_grad.append(get_auto_gradient(fi))
            fi_ineq_hess.append(get_auto_hessian(fi))
            s_max_init = max(s_max_init, fi(x_init))

        bm = BM.BarrierMethod(alpha=0.5, beta=0.1, sigma=1e-5, center_sigma=1e-5)

        x, s, _, _ = bm.solve_feasibility(f_ineq, fi_ineq_grad, fi_ineq_hess, x_init, None, None)

        s_max_sol = 0
        for fi in f_ineq:
            self.assertTrue(fi(x) <= 0)
            s_max_sol = max(s_max_sol, fi(x))

        print(f"s_max_init: {s_max_init} \ns_max_sol: {s_max_sol}")

    def test_infeasibility(self):
        fi_1 = self.create_affine_inequality(np.array([2 / 3, -1]), -2)
        fi_2 = self.create_affine_inequality(np.array([1, 1]), -4)
        fi_3 = self.create_affine_inequality(np.array([-1, 0]), 0)
        fi_ineq = [fi_1, fi_2, fi_3]
        fi_ineq_grad = []
        fi_ineq_hess = []
        for fi in fi_ineq:
            fi_ineq_grad.append(get_auto_gradient(fi))
            fi_ineq_hess.append(get_auto_hessian(fi))

        bm = BM.BarrierMethod(alpha=0.5, beta=0.1, sigma=1e-5, center_sigma=1e-5)

        x_init = np.array([100, 100])
        x, s, _, _ = bm.solve_feasibility(fi_ineq, fi_ineq_grad, fi_ineq_hess, x_init, None, None)

        feasible = True
        s_max_sol = 0
        for fi in fi_ineq:
            if(fi(x) > 0):
                feasible = False
            s_max_sol = max(s_max_sol, fi(x))

        self.assertFalse(feasible)
        print(f"s_max_sol: {s_max_sol}")


if __name__ == '__main__':
    unittest.main()
