import unittest
import numpy as np
from LineSearch import strong_wolfe_line_search, zoom

# --- Begin: Assume these functions are defined in your module ---
# For this example we assume the implementations are already in scope.
# If they are in a separate file (say, `line_search.py`), you could import them:
#
# from line_search import strong_wolfe_line_search, zoom
#
# Here is a reminder of their expected signatures:
#
# def strong_wolfe_line_search(x, delta_x, f, f_grad, a_max, c1, c2, a_growth=0.1, iteration=10):
#     ...
#
# def zoom(phi, phi_grad, a_lo, a_hi, c1, c2, iteration=10):
#     ...
#
# --- End: Assume strong_wolfe_line_search and zoom are available ---


class TestStrongWolfeLineSearch(unittest.TestCase):

    def setUp(self):
        # Define a simple quadratic function and its gradient.
        # Let f(x) = (x - 3)^2, so its minimum is at x = 3.
        # In one-dimension with x represented as a 1-element numpy array.
        self.f = lambda x: (x[0] - 3)**2
        self.f_grad = lambda x: np.array([2 * (x[0] - 3)])
        # Starting point
        self.x = np.array([0.0])
        # Descent direction (unit length) along which we want to search.
        # At x=0, f_grad(x) = 2*(0-3) = -6, so the direction [1.0] is descent.
        self.delta_x = np.array([1.0])
        # Wolfe parameters (typically 0 < c1 < c2 < 1)
        self.c1 = 1e-4
        self.c2 = 0.9
        # Maximum step length allowed
        self.a_max = 10.0
        # Tolerance for our test comparisons
        self.tol = 1e-6

    def test_strong_wolfe_quadratic(self):
        """Test that strong_wolfe_line_search finds a valid step for a quadratic function."""
        success, a_star = strong_wolfe_line_search(self.x, self.delta_x, self.f, self.f_grad,
                                                   self.a_max, self.c1, self.c2)
        self.assertTrue(success, "Line search failed on quadratic function.")

        # Define the one-dimensional functions phi(a) and phi_grad(a)
        # where phi(a) = f(x + a * delta_x)
        phi = lambda a: self.f(self.x + a * self.delta_x)
        phi_grad = lambda a: self.f_grad(self.x + a * self.delta_x) @ self.delta_x

        phi0 = phi(0)
        phi_grad0 = phi_grad(0)
        phi_a_star = phi(a_star)
        phi_grad_a_star = phi_grad(a_star)

        # Check the sufficient decrease (Armijo) condition:
        #   phi(a_star) <= phi(0) + c1 * a_star * phi_grad(0)
        self.assertLessEqual(
            phi_a_star,
            phi0 + self.c1 * a_star * phi_grad0 + self.tol,
            "Sufficient decrease condition not met."
        )

        # Check the curvature condition:
        #   |phi_grad(a_star)| <= c2 * |phi_grad(0)|
        self.assertLessEqual(
            np.abs(phi_grad_a_star),
            self.c2 * np.abs(phi_grad0) + self.tol,
            "Curvature condition not met."
        )

    def test_zoom_quadratic(self):
        """Test the zoom procedure directly on a quadratic function."""
        c1 = self.c1
        c2 = self.c2

        # Here we define phi and phi_grad corresponding to the quadratic function.
        # We choose the starting point for phi as if x = 0 and delta_x = 1, so that:
        #   phi(a) = (a - 3)^2 and phi_grad(a) = 2*(a - 3).
        phi = lambda a: (a - 3)**2
        phi_grad = lambda a: 2 * (a - 3)

        # For phi(0)=9 and phi_grad(0)=-6, choose a bracketing interval that contains the minimizer (at a=3).
        a_lo, a_hi = 1.0, 4.0
        success, a_star = zoom(phi, phi_grad, a_lo, a_hi, c1, c2, iteration=20)
        self.assertTrue(success, "Zoom failed on quadratic function.")

        phi0 = phi(0)
        phi_grad0 = phi_grad(0)

        self.assertLessEqual(
            phi(a_star),
            phi0 + c1 * a_star * phi_grad0 + self.tol,
            "Zoom: sufficient decrease condition not met."
        )
        self.assertLessEqual(
            np.abs(phi_grad(a_star)),
            c2 * np.abs(phi_grad0) + self.tol,
            "Zoom: curvature condition not met."
        )

    def test_no_valid_step(self):
        """Test that the line search fails when given a non-descent direction."""
        # For our quadratic function f(x) = (x-3)^2 at x = 0, f_grad(x) = -6.
        # If we choose delta_x = [-1], then f_grad(x) dot delta_x = (-6)*(-1) = 6, i.e. not a descent direction.
        x = np.array([0.0])
        delta_x = np.array([-1.0])
        success, a_star = strong_wolfe_line_search(x, delta_x, self.f, self.f_grad,
                                                   self.a_max, self.c1, self.c2)
        self.assertFalse(success, "Line search should fail for a non-descent direction.")


class TestStrongWolfeMultivariateQuadratic(unittest.TestCase):
    def setUp(self):
        # Define a 2D quadratic function: f(x) = 0.5 * ||x - x_opt||^2
        # with minimum at x_opt = [2, 2].
        self.x_opt = np.array([2.0, 2.0])
        self.f = lambda x: 0.5 * np.dot(x - self.x_opt, x - self.x_opt)
        self.f_grad = lambda x: x - self.x_opt

        # Choose a starting point x0 and compute a descent direction.
        self.x0 = np.array([0.0, 0.0])
        grad_x0 = self.f_grad(self.x0)  # For x0 = [0, 0], this is [-2, -2]
        # The descent direction is -grad(x0) normalized:
        self.delta_x = -grad_x0 / np.linalg.norm(grad_x0)

        # Wolfe parameters (0 < c1 < c2 < 1)
        self.c1 = 1e-4
        self.c2 = 0.9
        self.a_max = 10.0
        self.tol = 1e-6

    def test_strong_wolfe_conditions_multivariate(self):
        """Test that strong_wolfe_line_search finds a step that satisfies strong Wolfe conditions for a multivariate quadratic."""
        # Run the line search on the multivariate quadratic function.
        success, a_star = strong_wolfe_line_search(self.x0, self.delta_x,
                                                   self.f, self.f_grad,
                                                   self.a_max, self.c1, self.c2)
        self.assertTrue(success, "Line search failed for multivariate quadratic function.")

        # Define the one-dimensional function along the search direction:
        #   phi(a) = f(x0 + a * delta_x)
        phi = lambda a: self.f(self.x0 + a * self.delta_x)
        # And its directional derivative:
        phi_grad = lambda a: self.f_grad(self.x0 + a * self.delta_x) @ self.delta_x

        phi0 = phi(0)
        phi_grad0 = phi_grad(0)
        phi_a_star = phi(a_star)
        phi_grad_a_star = phi_grad(a_star)

        # Check the sufficient decrease (Armijo) condition:
        #   phi(a_star) <= phi(0) + c1 * a_star * phi_grad(0)
        self.assertLessEqual(
            phi_a_star,
            phi0 + self.c1 * a_star * phi_grad0 + self.tol,
            "Sufficient decrease condition not met for multivariate quadratic."
        )

        # Check the curvature condition:
        #   |phi_grad(a_star)| <= c2 * |phi_grad(0)|
        self.assertLessEqual(
            np.abs(phi_grad_a_star),
            self.c2 * np.abs(phi_grad0) + self.tol,
            "Curvature condition not met for multivariate quadratic."
        )


if __name__ == '__main__':
    unittest.main()
