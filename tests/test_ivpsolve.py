"""Tests for solve convenience function."""

import jax.numpy as jnp
import pytest

import tornado


@pytest.fixture(params=["ek1_ref", "ek1_diag"])
def solve_method(request):
    return request.param


@pytest.fixture(params=[2, 5])
def order(request):
    return request.param


@pytest.fixture
def dt():
    return 0.1


@pytest.fixture
def time_domain():
    return (0.0, 1.5)


def test_solve_constant(solve_method, order, time_domain, dt):

    t0, tmax = time_domain
    ivp = tornado.ivp.vanderpol(t0=0.0, tmax=1.5)

    with pytest.raises(KeyError):
        solution, solver = tornado.ivpsolve.solve(
            ivp,
            method="nonexisting",
            solver_order=order,
            adaptive=False,
            dt=dt,
            benchmark_mode=False,
        )

    solution, solver = tornado.ivpsolve.solve(
        ivp,
        method=solve_method,
        solver_order=order,
        adaptive=False,
        dt=dt,
        benchmark_mode=False,
    )

    expected_num_steps = int((tmax - t0) / dt) + 1
    assert len(solution.t) == len(solution.y) == expected_num_steps

    assert jnp.allclose(jnp.arange(t0, tmax + dt, step=dt), solution.t)

    for mean, cov_chol, cov in zip(solution.mean, solution.cov_sqrtm, solution.cov):
        try:
            cov = cov.todense()
            cov_chol = cov_chol.todense()
        except AttributeError:
            pass
        assert mean.shape == (ivp.dimension * (order + 1),)
        assert (solver.P0 @ mean).size == ivp.dimension
        assert cov.shape == (mean.shape[0], mean.shape[0])
        assert jnp.allclose(cov, cov_chol @ cov_chol.T)