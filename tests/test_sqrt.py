"""Tests for square-root utilities."""

import jax.numpy as jnp
import pytest

import tornado


@pytest.fixture
def iwp():
    return tornado.iwp.IntegratedWienerTransition(
        wiener_process_dimension=1, num_derivatives=1
    )


@pytest.fixture
def H_and_SQ(iwp, measurement_style):
    H, SQ = iwp.preconditioned_discretize_1d

    if measurement_style == "full":
        return H, SQ
    return H[:1], SQ[:1, :1]


@pytest.fixture
def SC(iwp):
    return iwp.preconditioned_discretize_1d[1]


@pytest.fixture
def batch_size():
    return 3


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_propagate_cholesky_factor(H_and_SQ, SC, measurement_style):
    H, SQ = H_and_SQ

    # First test: Non-optional S2
    chol = tornado.sqrt.propagate_cholesky_factor(S1=(H @ SC), S2=SQ)
    cov = H @ SC @ SC.T @ H.T + SQ @ SQ.T
    assert jnp.allclose(chol @ chol.T, cov)
    assert jnp.allclose(jnp.linalg.cholesky(cov), chol)
    assert jnp.all(jnp.diag(chol) > 0)


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_batched_propagate_cholesky_factors(
    H_and_SQ, SC, measurement_style, batch_size
):

    H, SQ = H_and_SQ
    H = tornado.linops.BlockDiagonal(jnp.stack([H] * batch_size))
    SQ = tornado.linops.BlockDiagonal(jnp.stack([SQ] * batch_size))
    SC = tornado.linops.BlockDiagonal(jnp.stack([SC] * batch_size))

    chol = tornado.sqrt.batched_propagate_cholesky_factor(
        (H @ SC).array_stack, SQ.array_stack
    )
    chol_as_bd = tornado.linops.BlockDiagonal(chol)
    reference = tornado.sqrt.propagate_cholesky_factor((H @ SC).todense(), SQ.todense())
    assert jnp.allclose(chol_as_bd.todense(), reference)


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_batched_sqrtm_to_cholesky(H_and_SQ, SC, measurement_style, batch_size):
    H, SQ = H_and_SQ
    d = H.shape[0]
    H = tornado.linops.BlockDiagonal(jnp.stack([H] * batch_size))
    SC = tornado.linops.BlockDiagonal(jnp.stack([SC] * batch_size))

    chol = tornado.sqrt.batched_sqrtm_to_cholesky((H @ SC).T.array_stack)
    chol_as_bd = tornado.linops.BlockDiagonal(chol)

    reference = tornado.sqrt.sqrtm_to_cholesky((H @ SC).T.todense())
    assert jnp.allclose(chol_as_bd.todense(), reference)
    assert chol_as_bd.array_stack.shape == (3, d, d)


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_update_sqrt(H_and_SQ, SC, measurement_style):
    """Test the square-root updates."""

    H, _ = H_and_SQ
    SC_new, kalman_gain, innov_chol = tornado.sqrt.update_sqrt(H, SC)
    assert isinstance(SC_new, jnp.ndarray)
    assert isinstance(kalman_gain, jnp.ndarray)
    assert isinstance(innov_chol, jnp.ndarray)
    assert SC_new.shape == SC.shape
    assert kalman_gain.shape == (H.shape[1], H.shape[0])
    assert innov_chol.shape == (H.shape[0], H.shape[0])

    # expected:
    S = H @ SC @ SC.T @ H.T
    K = SC @ SC.T @ H.T @ jnp.linalg.inv(S)
    C = SC @ SC.T - K @ S @ K.T

    # Test SC
    assert jnp.allclose(SC_new @ SC_new.T, C)
    assert jnp.allclose(SC_new, jnp.tril(SC_new))
    assert jnp.all(jnp.diag(SC_new) >= 0)

    # Test K
    assert jnp.allclose(K, kalman_gain)

    # Test S
    assert jnp.allclose(innov_chol @ innov_chol.T, S)
    assert jnp.allclose(innov_chol, jnp.tril(innov_chol))
    assert jnp.all(jnp.diag(innov_chol) >= 0)


def test_batched_update_sqrt(iwp):

    H, process_noise_cholesky = iwp.preconditioned_discretize_1d
    d = process_noise_cholesky.shape[0]
    for transition_matrix in [H, H[:1]]:
        A = tornado.linops.BlockDiagonal(jnp.stack([transition_matrix] * 3))
        some_chol = tornado.linops.BlockDiagonal(
            jnp.stack([process_noise_cholesky.copy()] * 3)
        )

        chol, K, S = tornado.sqrt.batched_update_sqrt(
            A.array_stack,
            some_chol.array_stack,
        )
        print(chol.shape, K.shape, S.shape)
        assert isinstance(chol, jnp.ndarray)
        assert isinstance(K, jnp.ndarray)
        assert isinstance(S, jnp.ndarray)
        assert K.shape == (3, d, transition_matrix.shape[0])
        assert chol.shape == (3, d, d)
        assert S.shape == (3, transition_matrix.shape[0], transition_matrix.shape[0])

        chol_as_bd = tornado.linops.BlockDiagonal(chol)
        K_as_bd = tornado.linops.BlockDiagonal(K)
        S_as_bd = tornado.linops.BlockDiagonal(S)
        ref_chol, ref_K, ref_S = tornado.sqrt.update_sqrt(
            A.todense(), some_chol.todense()
        )

        assert jnp.allclose(K_as_bd.todense(), ref_K)

        # The Cholesky-factor of positive semi-definite matrices is only unique
        # up to column operations (e.g. column reordering), i.e. there could be slightly
        # different Cholesky factors in batched and non-batched versions.
        # Therefore, we only check that the results are valid Cholesky factors themselves
        assert jnp.allclose((S_as_bd @ S_as_bd.T).todense(), ref_S @ ref_S.T)
        assert jnp.all(jnp.diag(S_as_bd.todense()) >= 0.0)
        assert jnp.allclose(
            (chol_as_bd @ chol_as_bd.T).todense(), ref_chol @ ref_chol.T
        )
        assert jnp.all(jnp.diag(chol_as_bd.todense()) >= 0.0)


def test_tril_to_positive_tril():
    """Assert that the weird sign(0)=0 behaviour is made up for."""
    matrix = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ]
    )
    result = tornado.sqrt.tril_to_positive_tril(matrix)
    assert jnp.allclose(matrix, result)
