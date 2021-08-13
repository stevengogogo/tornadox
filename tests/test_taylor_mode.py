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


class TestTaylorModeInitialization:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.taylor_init = tornado.taylor_mode.TaylorModeInitialization()

    def _construct_prior(self, order, spatialdim, t0):
        """Construct a prior process of appropriate size."""
        prior = tornado.iwp.IntegratedWienerTransition(
            wiener_process_dimension=spatialdim, num_derivatives=order
        )
        return prior

    @pytest.mark.parametrize("any_order", [0, 1, 2, 3])
    def test_call(self, any_order):
        threebody_ivp = tornado.ivp.threebody()

        prior = self._construct_prior(
            order=any_order, spatialdim=threebody_ivp.dimension, t0=threebody_ivp.t0
        )

        expected = prior.reorder_state_from_derivative_to_coordinate(
            THREEBODY_INITS[: threebody_ivp.dimension * (any_order + 1)]
        )

        received_rv = self.taylor_init(ivp=threebody_ivp, prior=prior)

        assert isinstance(received_rv, tornado.rv.MultivariateNormal)
        assert jnp.allclose(received_rv.mean, expected)
