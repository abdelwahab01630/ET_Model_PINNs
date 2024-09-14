import deepxde as dde
import numpy as np
import tensorflow as tf


class ETModel:
    def __init__(self,
                 delta: float = 0.5,
                 D: float = 1.0,
                 D_plus: float = 1.0,
                 K0: float = 20,
                 P1: float = -10,
                 P2: float = 10
                ):
        # Set constants
        self.delta = delta  # Electron transfer coefficient
        self.D = D  # Ratio of diffusion coefficient of A
        self.D_plus = D_plus  # Ratio of diffusion coefficient of B
        self.K0 = K0  # Dimensionles electron transfer rate constant
        self.P1 = P1  # Starting dimensionless potential
        self.P2 = P2  # End dimensionless potential

        self.D_depth = 5 * (2 * np.pi * (P2 + 8)) ** 0.5
        self.t_lambda = P2 - P1

    def dimensionless_potential(self, t):
        p = np.where(t <= self.t_lambda, self.P1 + t, self.P2 - (t - self.t_lambda))
        return p

    def electron_transfer_rates(self, P):
        Kf = self.K0 * np.exp((1 - self.delta) * P)
        Kb = self.K0 * np.exp((-self.delta) * P)
        return Kf, Kb

    def pde(self, x, y):
        cq_t = dde.grad.jacobian(y, x, i=0, j=1)
        cq_zz = dde.grad.hessian(y, x, i=0, j=0)
        r = cq_t - cq_zz
        return r

    def coupled_pdes(self, x, y):
        cq_t = dde.grad.jacobian(y, x, i=0, j=1)
        cqp_t = dde.grad.jacobian(y, x, i=1, j=1)
        cq_zz = dde.grad.hessian(y, x, component=0, i=0, j=0)
        cqp_zz = dde.grad.hessian(y, x, component=1, i=0, j=0)

        eq1 = cq_t - self.D*cq_zz
        eq2 = cqp_t - self.D_plus * cqp_zz
        return [eq1, eq2]

    def boundary_left(self, x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0.0)

    def boundary_right(self, x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], self.D_depth)

    def initial(self, x, on_initial):
        return on_initial and dde.utils.isclose(x[1], 0.0)
    
    def neumann_boundary_condition_cq(self, x):
        return 0.0

    def neumann_boundary_condition_cqp(self, x):
        return 0.0
    
    def robin_boundary_condition_cq(self, x, y):
        t = x[:, 1]
        cq = y[:, 0]
        cqp = y[:, 1]
        P = self.dimensionless_potential(t)
        k_f, k_b = self.electron_transfer_rates(P)
        r = (1.0 / self.D) * (k_f * cq - k_b * cqp)
        return r

    def robin_boundary_condition_cqp(self, x, y):
        t = x[:, 1]
        cq = y[:, 0]
        cqp = y[:, 1]
        P = self.dimensionless_potential(t)
        k_f, k_b = self.electron_transfer_rates(P)
        r = (1.0 / self.D_plus) * (-k_f * cq + k_b * cqp)
        return r

    def robin_boundary_condition(self, x, y):
        t = x[:, 1]
        P = self.dimensionless_potential(t)
        k_f, k_b = self.electron_transfer_rates(P)
        y = tf.reshape(y, shape=[-1])
        return (k_f + k_b)*y - k_b

    def dydx(self, x, y, X):
        return dde.grad.jacobian(y, x, i=0, j=0)
    
