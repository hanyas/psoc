from functools import partial
from typing import Tuple, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chex import Numeric, Array
from distrax import MultivariateNormalTri, Uniform
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.stats import norm
from jax.tree_util import tree_map


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_score_fn(seed):
    key = jax.random.PRNGKey(seed)
    n_iter = 5_000
    n_samples = 16

    np.random.seed(seed)
    F = np.random.randn(2, 2)
    b = np.random.randn(2)
    Q = np.random.rand(2, 5)
    Q = Q @ Q.T
    H = np.random.randn(1, 2)
    c = np.random.randn(1)
    R = np.random.rand(1, 1)

    ys = np.random.randn(1, 1) * np.ones((10, 1))  # repeat the same observation
    m0 = np.random.randn(2)
    P0 = np.zeros((2, 2))

    cholQ = np.linalg.cholesky(Q)


    dynamics = _StochasticDynamics(F, b, cholQ)
    transition_model = _RandomPolicy()
    log_observation = lambda x: norm.logpdf(ys[0, 0], x @ H.T + c, R[0, 0] ** 0.5)


    def loss_fn(x, xn, _):
        return dynamics.logpdf(x, None, xn) + log_observation(xn)

    def score_fn(x, xn, _):
        from jax import closure_convert

    def kf_loss(F_, b_, Q_, H_, c_, R_):
        ms, Ps, ell = filtering(ys, m0, P0, F_, Q_, b_, H_, R_, c_, update_first=False)
        return -ell

    ll, scores = jax.value_and_grad(kf_loss, argnums=(0, 1, 2, 3, 4, 5))(F, b, Q, H, c, R)

    print(scores)
    print(ll)


    # Testing the score function for increasing number of RB particles


class _StochasticDynamics(NamedTuple):
    F: Array
    b: Array
    cholQ: Array

    @property
    def dim(self):
        return self.F.shape[0]

    def sample(self, key, x, _u):
        dist = MultivariateNormalTri(loc=x @ self.F.T, scale_tri=self.cholQ)
        return dist.sample(seed=key)

    def logpdf(self, x, _u, xn):
        dist = MultivariateNormalTri(loc=x @ self.F.T, scale_tri=self.cholQ)
        return dist.log_prob(xn)


class _RandomPolicy(NamedTuple):
    @property
    def dim(self):
        return 0

    def sample(self, key, _x):
        dist = Uniform(low=-1., high=1.)
        return dist.sample(seed=key)

    def logpdf(self, _x, u):
        dist = Uniform(low=-1., high=1.)
        return dist.log_prob(u)


####################
# Kalman utilities #
####################
def filtering(ys, m0, P0, F, Q, b, H, R, c, update_first=False) -> Tuple[Array, Array, Numeric]:
    """
    Kalman filtering algorithm.
    If the number of observations is equal to the number of states, the first observation is used to update the initial state.
    Otherwise, the initial state is not updated but propagated first.

        Parameters
        ----------
        ys : Array
            Observations of shape (T, d_y).
        m0, P0, F, Q, b, H, R, c: Array
            LGSSM parameters.

        Returns
        -------
        ms : Array
            Filtered state means.
        Ps : Array
            Filtered state covariances.
        ell : Numeric
            Log-likelihood of the observations.
        """

    ell0 = 0.
    if update_first:
        # split between initial observation and rest
        y0, ys = ys[0], ys[1:]
        # Update initial state
        m0, P0, ell0 = sequential_update(y0, m0, P0, H, c, R)

    def body(carry, y):
        m, P, curr_ell = carry
        m, P, ell_inc = sequential_predict_update(m, P, F, b, Q, y, H, c, R)
        return (m, P, curr_ell + ell_inc), (m, P)

    (*_, ell), (ms, Ps) = jax.lax.scan(body,
                                       (m0, P0, ell0),
                                       ys)

    ms, Ps = tree_map(lambda z, y: jnp.insert(z, 0, y, axis=0), (ms, Ps), (m0, P0))
    return ms, Ps, ell


#                                   y,    m,     P,     H,    c,    R,  ->  m,     P,  ell
@partial(jnp.vectorize, signature='(dy),(dx),(dx,dx),(dy,dx),(dy),(dy,dy)->(dx),(dx,dx),()')
def sequential_update(y, m, P, H, c, R):
    y_hat = H @ m + c
    y_diff = y - y_hat

    S = R + H @ P @ H.T

    chol_S = jnp.linalg.cholesky(S)
    ell_inc = mvn_logpdf(y, y_hat, chol_S)
    G = cho_solve((chol_S, True), H @ P).T

    m = m + G @ y_diff

    P = P - G @ S @ G.T
    P = 0.5 * (P + P.T)
    return m, P, jnp.nan_to_num(ell_inc, nan=0.)


#                                   m,     P,      F,     b,    Q,  ->  m,    P,
@partial(jnp.vectorize, signature='(dx),(dx,dx),(dx,dx),(dx),(dx,dx)->(dx),(dx,dx)')
def sequential_predict(m, P, F, b, Q):
    m = F @ m + b
    P = Q + F @ P @ F.T
    P = 0.5 * (P + P.T)
    return m, P


#                                   m,     P,      F,     b,    Q,     y,    H,     c,    R   ->  m,    P,   ell
@partial(jnp.vectorize, signature='(dx),(dx,dx),(dx,dx),(dx),(dx,dx),(dy),(dy,dx),(dy),(dy,dy)->(dx),(dx,dx),()')
def sequential_predict_update(m, P, F, b, Q, y, H, c, R):
    m, P = sequential_predict(m, P, F, b, Q)
    m, P, ell_inc = sequential_update(y, m, P, H, c, R)
    return m, P, ell_inc


@partial(jnp.vectorize, signature="(n),(n),(n,n)->()")
def mvn_logpdf(x, m, chol):
    y = solve_triangular(chol, x - m, lower=True)
    norm_y = jnp.sum(y * y)
    return -0.5 * norm_y
