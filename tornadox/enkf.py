"""ENKF solvers."""

import dataclasses
from functools import cached_property, partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from tornado import init, ivp, iwp, linops, odesolver, rv, sqrt


@dataclasses.dataclass
class StateEnsemble:
    ivp: ivp.InitialValueProblem
    t: float
    samples: jnp.ndarray  # shape = [d * (nu + 1), N]

    @cached_property
    def ensemble_size(self):
        return self.samples.shape[1]

    @cached_property
    def dim(self):
        return self.samples.shape[0]

    @property
    def mean(self):
        return jnp.mean(self.samples, 1)

    @property
    def sample_cov(self):
        return jnp.cov(self.samples)


class EnK0(odesolver.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None
        self.ensemble_size = kwargs["ensemble_size"]

        self.rng = jax.random.PRNGKey(1)

    def initialize(self, ivp):
        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )

        self.P0 = self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        mean = extended_dy0.reshape((-1, 1), order="F")

        init_states = jnp.repeat(
            mean, self.ensemble_size, axis=1
        )  #  shape = [d * (nu + 1), N]
        assert init_states.shape == (
            ivp.dimension * (self.num_derivatives + 1),
            self.ensemble_size,
        )
        assert jnp.allclose(init_states[:, 0], mean.squeeze())
        assert jnp.allclose(init_states[:, -1], mean.squeeze())
        return StateEnsemble(ivp=ivp, t=ivp.t0, samples=init_states)

    def attempt_step(self, ensemble, dt, verbose=False):

        t_new = ensemble.t + dt

        # [Setup]
        A, Ql = self.iwp.non_preconditioned_discretize(dt)

        # [Predict]
        w = jax.random.multivariate_normal(
            self.rng,
            mean=jnp.zeros((ensemble.dim,)),
            cov=Ql @ Ql.T,
            shape=(ensemble.ensemble_size,),
        ).T

        _, self.rng = jax.random.split(self.rng)

        pred_samples = A @ ensemble.samples + w
        sample_mean = jnp.mean(pred_samples, axis=1)
        sample_cov = jnp.cov(pred_samples)

        H, _, b = self.evaluate_ode(
            t_new,
            ensemble.ivp.f,
            ensemble.ivp.df,
            sample_mean,
            self.E0,
            self.E1,
        )

        z_samples = H @ pred_samples + b[:, None]

        # Estimate Kalman gain
        # via Eq. (11) in https://www.math.umd.edu/~slud/RITF17/enkf-tutorial.pdf
        CHT = sample_cov @ H.T
        to_invert = H @ CHT
        gain_times_z = CHT @ jnp.linalg.solve(to_invert, z_samples)

        # Update
        updated_samples = pred_samples - gain_times_z

        return StateEnsemble(ivp=ensemble.ivp, t=t_new, samples=updated_samples)

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def evaluate_ode(t, f, df, m_pred, e0, e1):
        P0 = e0
        P1 = e1
        m_at = P0 @ m_pred
        f = f(t, m_at)
        Jx = df(t, m_at)
        H = P1 - Jx @ P0
        b = Jx @ m_at - f
        z = H @ m_pred + b
        return H, z, b

    @staticmethod
    def estimate_error(h, sq, z):
        s_sqrtm = h @ sq
        s_chol = sqrt.sqrtm_to_cholesky(s_sqrtm.T)

        whitened_res = jax.scipy.linalg.solve_triangular(s_chol.T, z, lower=False)
        sigma_squared = whitened_res.T @ whitened_res / whitened_res.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error_estimate = sigma * jnp.sqrt(jnp.diag(s_chol @ s_chol.T))
        return error_estimate, sigma
