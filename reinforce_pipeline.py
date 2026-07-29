import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax.tree_util import tree_map
import optax
import optax.tree_utils as otu
from functools import partial
import numpy as np

# optax renamed `tree_l2_norm` -> `tree_norm` in v0.2.5; support both so this runs
# on older and newer optax without a deprecation warning or an AttributeError.
_tree_l2_norm = getattr(otu, "tree_norm", None) or otu.tree_l2_norm

import matplotlib.pyplot as plt
from utils import get_nonlinearity, make_train_mask, flatten_params, generate_randomized_multisession_graded_data
from globals import DEPTH_TO_NUM_WEIGHTS, MU_ALPHA, MU_BETA

NUM_WEIGHTS = 3          # w = (bias, right, left) into a single output unit

# model name -> (num_alphas, num_baselines)
REINFORCE_MODELS = {
    'RF_1':   (1, 0),    # one learning rate, no baseline
    'RF_b_1': (1, 1),    # one learning rate, one baseline
    'RF':     (3, 0),    # per-weight learning rates, no baseline
    'RF_b':   (3, 3),    # per-weight learning rates and baselines
}


# INFERENCE UTILS 
# ---------------------------------------------------------------------------------------------- #

def get_num_params(model):
    """Total inferred parameters for `model` (use for AIC and for sizing inits)."""
    assert model in REINFORCE_MODELS, f'unknown model {model!r}'
    num_alphas, num_baselines = REINFORCE_MODELS[model]
    return num_alphas + num_baselines + NUM_WEIGHTS


@partial(jit, static_argnames=('model',))
def unpack_reinforce_hyperparams(hyperparams, model):
    """Split the flat parameter vector into (alphas, baselines, w0).

    Layout -- w0 is ALWAYS last, so `hyperparams[-NUM_WEIGHTS:]` is w0, matching the
    ridge-slicing convention already used in inference_pipeline.compute_nll:

        RF_1     [log_a,                  w0(3)]  ->  4 params
        RF_b_1   [log_a,      b,          w0(3)]  ->  5 params
        RF       [log_a(3),               w0(3)]  ->  6 params
        RF_b     [log_a(3),   b(3),       w0(3)]  ->  9 params

    Learning rates are inferred in LOG space (so alpha > 0 always); baselines are
    inferred directly, since they may be negative.

    Arguments:
    - hyperparams : 1-D array of length get_num_params(model)
    - model       : key into REINFORCE_MODELS (static under jit)

    Returns:
    - alphas    : (3,) learning rates, ALREADY exponentiated, broadcast from a
                  scalar for the single-rate models
    - baselines : (3,) reward baselines, broadcast from a scalar, and identically
                  zero for the no-baseline models (so one update expression covers
                  every variant)
    - w0        : (3,) initial weights (bias, right, left)
    """
    assert model in REINFORCE_MODELS, f'unknown model {model!r}; choose from {sorted(REINFORCE_MODELS)}'
    num_alphas, num_baselines = REINFORCE_MODELS[model]
    expected = num_alphas + num_baselines + NUM_WEIGHTS
    assert hyperparams.shape == (expected,), \
        f'{model} expects {expected} parameters, got shape {hyperparams.shape}'

    alphas = jnp.exp(hyperparams[:num_alphas])
    baselines = hyperparams[num_alphas:num_alphas + num_baselines]
    w0 = hyperparams[-NUM_WEIGHTS:]

    # broadcast to per-weight vectors so the update rule is a single expression
    if num_alphas == 1:
        alphas = jnp.repeat(alphas, NUM_WEIGHTS)
    if num_baselines == 0:
        baselines = jnp.zeros(NUM_WEIGHTS, dtype=hyperparams.dtype)
    elif num_baselines == 1:
        baselines = jnp.repeat(baselines, NUM_WEIGHTS)

    return alphas, baselines, w0


@jit
def log_likelihood(out, choice):
    '''
    out        : the model's output value
    choice     : the choice made by the agent (L = 0, R = 1 here!)
    '''
    ll = choice * out - jnp.logaddexp(0, out)
    return ll


# INFERENCE PIPELINE 
# ---------------------------------------------------------------------------------------------- #


@partial(jit, static_argnames=('model',))
def run_train_test_single_reinforce_forward_trajectory(hyperparams, model, inputs, outputs, rewards, train_mask):
    """Simulate the online-GD learning trajectory for a single sequence and split
    its negative log-likelihood into train and test portions via `train_mask`.

    Arguments:
    - hyperparams: 1-D array packed as (log_alphas, bs, w0)
    - model: specific REINFORCE learning model variant
    - inputs
    - outputs   (here 0 = left and 1 = right)
    - rewards
    - train_mask: boolean array of shape (seq_len,); True = train, False = test

    Returns:
    - train_nll: negative log-likelihood summed over training time steps
    - test_nll:  negative log-likelihood summed over held-out time steps
    - ws:        weight trajectory (pytree, leading axis = time)
    - lls:       per-trial log-likelihoods (before masking)
    """

    alphas, baselines, w0 = unpack_reinforce_hyperparams(hyperparams, model)

    def _step(w, data_point):
        x, y, rew = data_point

        # compute current model output
        out = jnp.dot(w, x)

        # compute probability of going right
        ll_R = log_likelihood(out, choice=1)
        p_R = jnp.exp(ll_R)

        # compute policy gradient
        update = (y - p_R) * (rew - baselines) * x

        # now, apply REINFORCE update
        w_new = w + alphas * update

        # for the current weights compute the log-likelihood of the data given the inference params 
        ll = log_likelihood(out, y)

        return w_new, (ll, w_new)

    data_seq = (inputs, outputs, rewards)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    # total_nll = -jnp.sum(lls)
    train_nll = -jnp.sum(jnp.where(train_mask,  lls, 0.0))
    test_nll  = -jnp.sum(jnp.where(~train_mask, lls, 0.0))

    return train_nll, test_nll, ws, lls


@partial(jit, static_argnames=('model',))
def run_single_learning_forward_trajectory(hyperparams, model, inputs, outputs, rewards):
    """Simulate the online-GD learning trajectory for a single sequence and return
    its total negative log-likelihood over all trials.

    Arguments:
    - hyperparams: 1-D array packed as (log_alphas, baselines, w0)
    - model: specific type of REINFORCE learning model
    - inputs
    - outputs   (L = 0 and R = 1 here!)
    - rewards

    Returns:
    - total_nll: negative log-likelihood summed over all time steps
    - ws:        weight trajectory (pytree, leading axis = time)
    - lls:       per-trial log-likelihoods
    """

    total_nll, _, ws, lls = run_train_test_single_reinforce_forward_trajectory(hyperparams, model, inputs, outputs, rewards, jnp.ones_like(outputs, dtype=bool))

    return total_nll, ws, lls


@partial(jit, static_argnames=('model', 'masked','ridge_lambda', 'ridge_alpha', 'ridge_b', 'mu_alpha', 'mu_b'))
def compute_objective(hyperparams, model, inputs, outputs, rewards,
                masked=False, train_mask=None, 
                ridge_lambda=0., ridge_alpha=0., ridge_b=0., mu_alpha=0., mu_b=0.):
    N = len(hyperparams)
    num_alphas, num_baselines = get_num_params(model)
    nll = None
    # `masked` is a static (compile-time) flag, so we pick the branch at trace time
    # with a plain Python if/elif/else. This avoids lax.cond tracing the train-mask
    # branch when train_mask is None (the plain single-animal case).
    if masked:
        # train/test split: score only the training trials (train_mask: True = train)
        nll = run_train_test_single_reinforce_forward_trajectory(hyperparams, model, inputs, outputs, rewards, train_mask)[0]
    else:
        nll = run_single_learning_forward_trajectory(hyperparams, model, inputs, outputs, rewards)[0]

    l2_prior = ridge_lambda * jnp.sum(hyperparams[-NUM_WEIGHTS:]**2) + ridge_alpha * jnp.sum((hyperparams[:num_alphas] - mu_alpha)**2) + ridge_b * jnp.sum((hyperparams[num_alphas:num_alphas + num_baselines] - mu_b)**2)
    objective = nll + l2_prior

    return objective

