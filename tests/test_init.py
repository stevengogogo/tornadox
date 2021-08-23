"""Tests for Taylor-mode initialization."""
import jax.numpy as jnp
import pytest

import tornado

# The usual initial values and parameters for the three-body problem
THREEBODY_INITS = jnp.array(
    [
        0.9939999999999999946709294817992486059665679931640625,
        0.0,
        0.0,
        -2.001585106379082379390865753521211445331573486328125,
        0.0,
        -2.001585106379082379390865753521211445331573486328125,
        -315.5430234888826712053567300105174012152470262808609555489430375415196001267175,
        0.0,
        -315.5430234888826712053567300105174012152470262808609555489430375415196001267175,
        0.0,
        0.0,
        99972.09449511380974582623407652494237674536945822956356849412439883437128612817,
        0.0,
        99972.09449511380974582623407652494237674536945822956356849412439883437128612817,
        6.390281114012432978693829866143426861527192861569087503897123013405669119845963e07,
        0.0,
        6.390281114012432978693829866143426861527192861569087503897123013405669119845963e07,
        0.0,
        0.0,
        -5.104537695521316959384194278813460762798119492148033782222345167969699208514969e10,
        0.0,
        -5.104537695521316959384194278813460762798119492148033782222345167969699208514969e10,
        -5.718989915866635673742953579223755513260251013224240606810015106258694103361678e13,
        0.0,
        -5.718989915866635673742953579223755513260251013224240606810015106258694103361678e13,
        0.0,
        0.0,
        7.315561441063621135318644202108449833826961384616926386892170458762613581663705e16,
        0.0,
        7.315561441063621135318644202108449833826961384616926386892170458762613581663705e16,
        1.171034721872789800168691106581608116625156325498263537591690807229212643359755e20,
        0.0,
        1.171034721872789800168691106581608116625156325498263537591690807229212643359755e20,
        0.0,
        0.0,
        -2.060304783152864457766016457053004312347844856370942449557112224304453885256121e23,
        0.0,
        -2.060304783152864457766016457053004312347844856370942449557112224304453885256121e23,
        -4.287443879083103146988750929545238409085724761354466323771047346244918232948406e26,
        0.0,
        -4.287443879083103146988750929545238409085724761354466323771047346244918232948406e26,
        0.0,
        0.0,
        9.601981786174386486553049044619090977452789641463027654594822277483844742280914e29,
        0.0,
        9.601981786174386486553049044619090977452789641463027654594822277483844742280914e29,
        2.45921764824840811167266438750049754390612890829237569709019616087539546093443e33,
        0.0,
        2.45921764824840811167266438750049754390612890829237569709019616087539546093443e33,
        0.0,
        0.0,
        -6.68835380913736620822410435219200785043714506466369418247249547659103361618871e36,
        0.0,
        -6.68835380913736620822410435219200785043714506466369418247249547659103361618871e36,
        -2.034138186021457890521425446714091788329533957393534339124615127934059591480461e40,
        0.0,
        -2.034138186021457890521425446714091788329533957393534339124615127934059591480461e40,
        0.0,
        0.0,
        6.50892954321055682909266276268766148490204239070955567967642805425258366069523e43,
        0.0,
        6.50892954321055682909266276268766148490204239070955567967642805425258366069523e43,
        2.292092276850920428783678416298069135189529108299808020118152953566355216581644e47,
        0.0,
    ]
).flatten()


@pytest.fixture
def ivp():
    return tornado.ivp.threebody()


@pytest.fixture
def nordsieck_y0(ivp, num_derivatives):
    return tornado.init.taylor_mode(
        fun=ivp.f, y0=ivp.y0, t0=ivp.t0, num_derivatives=num_derivatives
    )


@pytest.mark.parametrize("num_derivatives", [0, 1, 3])
def test_taylor_mode_shape(nordsieck_y0, ivp, num_derivatives):
    assert nordsieck_y0.shape == (num_derivatives + 1, ivp.dimension)


@pytest.mark.parametrize("num_derivatives", [3])
def test_taylor_mode_first_value_is_y0(nordsieck_y0, ivp):
    """Check the ordering is correct."""
    assert jnp.allclose(nordsieck_y0[0], ivp.y0)


@pytest.fixture
def threebody_nordsieck_initval():
    threebody_dimension = 4
    return THREEBODY_INITS.reshape((-1, threebody_dimension), order="C")


@pytest.mark.parametrize("num_derivatives", [3])
def test_taylor_expected_values(
    nordsieck_y0, threebody_nordsieck_initval, num_derivatives
):
    """Check the ordering is correct."""
    assert jnp.allclose(
        nordsieck_y0, threebody_nordsieck_initval[: (num_derivatives + 1)]
    )


# Tests for RK init


@pytest.fixture
def ivp2():
    return tornado.ivp.vanderpol(stiffness_constant=10, t0=0.0, tmax=30.0)
    # return tornado.ivp.threebody()


@pytest.fixture
def t0(ivp2):
    return ivp2.t0


@pytest.fixture
def y0(ivp2):
    return ivp2.y0


@pytest.fixture
def f(ivp2):
    return ivp2.f


@pytest.fixture
def df(ivp2):
    return ivp2.df


@pytest.fixture
def d(ivp2):
    return ivp2.dimension


@pytest.fixture
def dt():
    return 0.01


@pytest.fixture
def num_derivatives():
    return 4


@pytest.fixture
def num_steps(num_derivatives):
    return 2 * num_derivatives + 1


@pytest.fixture
def n(num_derivatives):
    return num_derivatives + 1


all_rk_methods = pytest.mark.parametrize("method", ["DOP853"])


@pytest.fixture
def rk_data(f, y0, t0, dt, num_steps, method):
    return tornado.init.rk_data(
        f=f, t0=t0, dt=dt, num_steps=num_steps, y0=y0, method=method
    )


@all_rk_methods
def test_rk_init_generate_data_types(rk_data):
    ts, ys = rk_data
    assert isinstance(ts, jnp.ndarray)
    assert isinstance(ys, jnp.ndarray)


@all_rk_methods
def test_rk_init_generate_data_shapes(rk_data, num_steps, d):
    ts, ys = rk_data
    assert ts.shape == (num_steps,)
    assert ys.shape == (num_steps, d)


@pytest.fixture
def init_stack(f, df, y0, t0, num_derivatives):
    return tornado.init.stack_initial_state_jac(
        f=f, df=df, y0=y0, t0=t0, num_derivatives=num_derivatives
    )


@pytest.fixture
def rk_init_improved(init_stack, t0, rk_data):
    m, sc = init_stack
    ts, ys = rk_data
    return tornado.init.rk_init_improve(
        m=m,
        sc=sc,
        t0=t0,
        ts=ts,
        ys=ys,
    )


@all_rk_methods
def test_rk_init_types(rk_init_improved):
    m, sc = rk_init_improved
    assert isinstance(m, jnp.ndarray)
    assert isinstance(sc, jnp.ndarray)


@all_rk_methods
def test_rk_init_shapes(rk_init_improved, n, d):
    m, sc = rk_init_improved

    assert m.shape == (n, d)
    assert sc.shape == (n, n)

    assert isinstance(m, jnp.ndarray)
    assert isinstance(sc, jnp.ndarray)


@pytest.fixture
def ref_init(f, y0, t0, num_derivatives):
    return tornado.init.taylor_mode(
        fun=f, y0=y0, t0=t0, num_derivatives=num_derivatives
    )


@all_rk_methods
def test_rk_init_values(rk_init_improved, ref_init):

    # Relaxed tolerance, because initialisation is only for ballpark estimates
    # The current values are rather sharp
    assert jnp.allclose(rk_init_improved[0], ref_init, rtol=1e-1, atol=1e-10)


def test_stack_initial_state_jac(f, df, y0, t0, num_derivatives):
    m0, sc0 = tornado.init.stack_initial_state_jac(
        f=f, df=df, y0=y0, t0=t0, num_derivatives=num_derivatives
    )

    assert m0.shape == (num_derivatives + 1, y0.shape[0])
    assert sc0.shape == (num_derivatives + 1, num_derivatives + 1)
