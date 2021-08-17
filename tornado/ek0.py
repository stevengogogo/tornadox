import dataclasses

import jax.numpy as jnp

from tornado import ivp, iwp, odesolver, rv, sqrt, step, taylor_mode


class ReferenceEK0(odesolver.ODEFilter):
    def initialize(self, ivp):
        d = ivp.dimension
        q = self.solver_order
        self.iwp = iwp.IntegratedWienerTransition(
            wiener_process_dimension=d, num_derivatives=q
        )
        Y0_full = taylor_mode.TaylorModeInitialization()(ivp, self.iwp)

        self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)
        self.I = jnp.eye(d * (q + 1))

        return odesolver.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            error_estimate=None,
            reference_state=ivp.y0,
            y=Y0_full,
        )

    def attempt_step(self, state, dt, verbose=False):
        # [Setup]
        m, Cl = state.y.mean, state.y.cov_cholesky
        A, Ql = self.iwp.non_preconditioned_discretize(dt)

        # [Predict]
        mp = A @ m
        Clp = sqrt.propagate_cholesky_factor(A @ Cl, Ql)

        # [Measure]
        z = self.E1 @ mp - state.ivp.f(state.t + dt, self.E0 @ mp)
        H = self.E1
        # Sl = H @ Clp
        # S = Sl @ Sl.T

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp)
        # K = (Clp @ Clp.T) @ H.T @ jnp.linalg.inv(S)
        m_new = mp - K @ z
        # Cl_new = (self.I - K @ H) @ Clp

        y_new = self.E0 @ m_new

        return odesolver.ODEFilterState(
            ivp=state.ivp,
            t=state.t + dt,
            error_estimate=None,
            reference_state=y_new,
            y=rv.MultivariateNormal(m_new, Cl_new),
        )


class EK0(odesolver.ODEFilter):
    def initialize(self, ivp):
        self.d = ivp.dimension
        self.q = self.solver_order
        self.iwp = iwp.IntegratedWienerTransition(
            wiener_process_dimension=self.d, num_derivatives=self.q
        )
        self.A, self.Ql = self.iwp.preconditioned_discretize_1d

        Y0_full = taylor_mode.TaylorModeInitialization()(ivp, self.iwp)
        Y0_kron = rv.MultivariateNormal(
            Y0_full.mean, jnp.zeros((self.q + 1, self.q + 1))
        )

        self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)
        self.e0 = self.iwp.projection_matrix_1d(0)
        self.e1 = self.iwp.projection_matrix_1d(1)
        self.Id = jnp.eye(self.d)
        self.Iq1 = jnp.eye(self.q + 1)

        return odesolver.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            error_estimate=None,
            reference_state=ivp.y0,
            y=Y0_kron,
        )

    def attempt_step(self, state, dt, verbose=False):
        # [Setup]
        Y = state.y
        _m, _Cl = y.mean, Y.cov_cholesky
        A, Ql = self.A, self.Ql

        t_new = state.t + dt

        # [Preconditioners]
        P, PI = self.iwp.nordsieck_preconditioner_1d(dt)
        m, Cl = vec_trick_mul_right(PI, _m), PI @ _Cl

        # [Predict]
        mp = vec_trick_mul_right(A, m)

        # [Measure]
        _mp = vec_trick_mul_right(P, mp)  # Undo the preconditioning
        z = self.E1 @ _mp - state.ivp.f(t_new, self.E0 @ _mp)
        H = self.e1 @ P

        # [Calibration]
        HQH = (H @ Ql @ Ql.T @ H.T)[0, 0]
        # HQH = Q11(dt)  # Q(dt)[1, 1]
        sigma_squared = z.T @ z / HQH / self.d
        # sigma_squared = 1.0

        # [Predict Covariance]
        Clp = sqrt.propagate_cholesky_factor(A @ Cl, jnp.sqrt(sigma_squared) * Ql)

        # [Update]
        # K = Clp @ Clp.T @ H.T / S
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp)
        m_new = mp - vec_trick_mul_right(K, z)
        Cl_new = (self.Iq1 - K @ H) @ Clp

        # [Undo preconditioning]
        _m_new, _Cl_new = vec_trick_mul_right(P, m_new), P @ Cl_new

        y_new = self.E0 @ _m_new

        error_estimate = (
            jnp.repeat(jnp.sqrt(sigma_squared * HQH), self.d)
            if isinstance(self.steprule, step.AdaptiveSteps)
            else None
        )

        return odesolver.ODEFilterState(
            ivp=state.ivp,
            t=t_new,
            error_estimate=error_estimate,
            reference_state=y_new,
            y=rv.MultivariateNormal(_m_new, _Cl_new),
        )


def vec_trick_mul_full(K1, K2, v):
    """Use the vec trick to compute kron(K1,K2)@v more efficiently"""
    (d1, d2), (d3, d4) = K1.shape, K2.shape
    V = v.reshape(d4, d2, order="F")
    return (K2 @ V @ K1.T).reshape(d1 * d3, order="F")


def vec_trick_mul_right(K2, v):
    """Use the vec trick to compute kron(I_d,K2)@v more efficiently"""
    d3, d4 = K2.shape
    V = v.reshape(d4, v.size // d4, order="F")
    out = K2 @ V
    return out.reshape(out.size, order="F")
