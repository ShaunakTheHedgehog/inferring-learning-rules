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

import argparse
import os
import pickle as pkl
import time

# optax renamed `tree_l2_norm` -> `tree_norm` in v0.2.5; support both so this runs
# on older and newer optax without a deprecation warning or an AttributeError.
_tree_l2_norm = getattr(otu, "tree_norm", None) or otu.tree_l2_norm

import matplotlib.pyplot as plt
from utils import (get_nonlinearity, make_train_mask, flatten_params,
                   generate_randomized_multisession_graded_data,
                   flip_outputs, check_writable, to_numpy)
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
def unpack_RF_hyperparams(hyperparams, model):
    """Split the flat parameter vector into (alphas, intercepts, w0).

    Layout -- w0 is ALWAYS last, so `hyperparams[-NUM_WEIGHTS:]` is w0, matching the
    ridge-slicing convention already used in inference_pipeline.compute_nll:

        RF_1     [log_a,                  w0(3)]  ->  4 params
        RF_b_1   [log_a,      c,          w0(3)]  ->  5 params
        RF       [log_a(3),               w0(3)]  ->  6 params
        RF_b     [log_a(3),   c(3),       w0(3)]  ->  9 params

    Learning rates are inferred in LOG space (so alpha > 0 always); intercepts are
    inferred directly, since they may be negative.

    Arguments:
    - hyperparams : 1-D array of length get_num_params(model)
    - model       : key into REINFORCE_MODELS (static under jit)

    Returns:
    - alphas    : (3,) learning rates, ALREADY exponentiated, broadcast from a
                  scalar for the single-rate models
    - intercepts : (3,) reward intercepts (alpha * baseline), broadcast from a scalar, and identically
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
    intercepts = hyperparams[num_alphas:num_alphas + num_baselines]
    w0 = hyperparams[-NUM_WEIGHTS:]

    # broadcast to per-weight vectors so the update rule is a single expression
    if num_alphas == 1:
        alphas = jnp.repeat(alphas, NUM_WEIGHTS)
    if num_baselines == 0:
        intercepts = jnp.zeros(NUM_WEIGHTS, dtype=hyperparams.dtype)
    elif num_baselines == 1:
        intercepts = jnp.repeat(intercepts, NUM_WEIGHTS)

    return alphas, intercepts, w0


@jit
def RF_log_likelihood(out, choice):
    '''
    out        : the model's output value
    choice     : the choice made by the agent (L = 0, R = 1 here!)
    '''
    ll = choice * out - jnp.logaddexp(0, out)
    return ll


def generate_RF_init_hyperparams(model, alpha_init_scale, c_init_scale, w0_init_scale, key,
                              alpha_init_guess=None, c_init_guess=None, w0_init_guess=None,
                              noise_level=0.01):
    '''
    Draw a random starting point for one optimization seed, packed in the layout that
    `unpack_reinforce_hyperparams` expects: [log_alphas, intercepts, w0].

    Learning rates are drawn positive (half-normal) and then logged, since they are
    inferred in log space. Intercepts are drawn from a ZERO-MEAN normal: c = 0 makes the
    update rule identical to plain REINFORCE, so every intercept model starts at its
    no-intercept special case and only moves away if the data support it. Initial weights
    are likewise zero-mean normal.

    Arguments:
    - model            : key into REINFORCE_MODELS
    - alpha_init_scale : scale of the half-normal the learning rates are drawn from
    - c_init_scale     : s.d. of the zero-mean normal the intercepts are drawn from
    - w0_init_scale    : s.d. of the zero-mean normal the initial weights are drawn from
    - key              : JAX PRNG key
    - *_init_guess     : optionally start near a known value instead of drawing fresh.
                         alpha/w0 are perturbed multiplicatively (as in inference_utils),
                         but the intercept guess is perturbed ADDITIVELY, since a
                         multiplicative jitter around a guess of 0 would never move.

    Returns:
    - 1-D array of length get_num_params(model)
    '''
    assert model in REINFORCE_MODELS, f'unknown model {model!r}; choose from {sorted(REINFORCE_MODELS)}'
    num_alphas, num_baselines = REINFORCE_MODELS[model]
    subkeys = jr.split(key, 6)

    alpha_inits = alpha_init_scale * jnp.abs(jr.normal(subkeys[1], (num_alphas,)))
    if alpha_init_guess is not None:
        assert jnp.all(jnp.asarray(alpha_init_guess) > 0.0)
        alpha_inits = alpha_init_guess * (1 + noise_level * jr.normal(subkeys[2], (num_alphas,)))
    log_alpha_inits = jnp.log(alpha_inits)

    # zero-mean: c = 0 is plain REINFORCE, so the intercept models start there
    c_inits = c_init_scale * jr.normal(subkeys[3], (num_baselines,))
    if c_init_guess is not None:
        c_inits = c_init_guess + noise_level * jr.normal(subkeys[4], (num_baselines,))

    w0 = w0_init_scale * jr.normal(subkeys[5], (NUM_WEIGHTS,))
    if w0_init_guess is not None:
        w0 = w0_init_guess * (1 + noise_level * jr.normal(subkeys[5], (NUM_WEIGHTS,)))

    init_hyperparams_arr = jnp.concatenate([log_alpha_inits, c_inits, w0])
    return init_hyperparams_arr


# INFERENCE PIPELINE 
# ---------------------------------------------------------------------------------------------- #


@partial(jit, static_argnames=('model',))
def run_train_test_single_RF_forward_trajectory(hyperparams, model, inputs, outputs, rewards, train_mask):
    """Simulate the online-GD learning trajectory for a single sequence and split
    its negative log-likelihood into train and test portions via `train_mask`.

    Arguments:
    - hyperparams: 1-D array packed as (log_alphas, intercepts, w0)
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

    alphas, intercepts, w0 = unpack_RF_hyperparams(hyperparams, model)

    def _step(w, data_point):
        x, y, rew = data_point

        # compute current model output
        out = jnp.dot(w, x)

        # compute probability of going right
        ll_R = RF_log_likelihood(out, choice=1)
        p_R = jnp.exp(ll_R)

        # compute policy gradient
        update = (y - p_R) * (alphas * rew - intercepts) * x

        # now, apply REINFORCE update
        w_new = w + update

        # for the current weights compute the log-likelihood of the data given the inference params 
        ll = RF_log_likelihood(out, y)

        return w_new, (ll, w_new)

    data_seq = (inputs, outputs, rewards)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    # total_nll = -jnp.sum(lls)
    train_nll = -jnp.sum(jnp.where(train_mask,  lls, 0.0))
    test_nll  = -jnp.sum(jnp.where(~train_mask, lls, 0.0))

    return train_nll, test_nll, ws, lls


@partial(jit, static_argnames=('model',))
def run_single_RF_learning_forward_trajectory(hyperparams, model, inputs, outputs, rewards):
    """Simulate the online-GD learning trajectory for a single sequence and return
    its total negative log-likelihood over all trials.

    Arguments:
    - hyperparams: 1-D array packed as (log_alphas, intercepts, w0)
    - model: specific type of REINFORCE learning model
    - inputs
    - outputs   (L = 0 and R = 1 here!)
    - rewards

    Returns:
    - total_nll: negative log-likelihood summed over all time steps
    - ws:        weight trajectory (pytree, leading axis = time)
    - lls:       per-trial log-likelihoods
    """

    total_nll, _, ws, lls = run_train_test_single_RF_forward_trajectory(hyperparams, model, inputs, outputs, rewards, jnp.ones_like(outputs, dtype=bool))

    return total_nll, ws, lls


@partial(jit, static_argnames=('model', 'masked','ridge_lambda', 'ridge_alpha', 'ridge_c', 'mu_alpha', 'mu_c'))
def compute_RF_objective(hyperparams, model, inputs, outputs, rewards,
                masked=False, train_mask=None, 
                ridge_lambda=0., ridge_alpha=0., ridge_c=0., mu_alpha=0., mu_c=0.):
    N = len(hyperparams)
    num_alphas, num_baselines = REINFORCE_MODELS[model]
    nll = None
    # `masked` is a static (compile-time) flag, so we pick the branch at trace time
    # with a plain Python if/elif/else. This avoids lax.cond tracing the train-mask
    # branch when train_mask is None (the plain single-animal case).
    if masked:
        # train/test split: score only the training trials (train_mask: True = train)
        nll = run_train_test_single_RF_forward_trajectory(hyperparams, model, inputs, outputs, rewards, train_mask)[0]
    else:
        nll = run_single_RF_learning_forward_trajectory(hyperparams, model, inputs, outputs, rewards)[0]

    l2_prior = ridge_lambda * jnp.sum(hyperparams[-NUM_WEIGHTS:]**2) + ridge_alpha * jnp.sum((hyperparams[:num_alphas] - mu_alpha)**2) + ridge_c * jnp.sum((hyperparams[num_alphas:num_alphas + num_baselines] - mu_c)**2)
    objective = nll + l2_prior

    return objective


@partial(jit, static_argnames=('model', 'masked', 'ridge_lambda', 'ridge_alpha', 'ridge_c', 'mu_alpha', 'mu_c'))
def compute_RF_hessian(hyperparams, model, inputs, outputs, rewards,
                    masked=False, train_mask=None,
                    ridge_lambda=0., ridge_alpha=0., ridge_c=0., mu_alpha=0., mu_c=0.):
    hess_fn = jax.hessian(compute_RF_objective)
    hessian = hess_fn(hyperparams, model, inputs, outputs, rewards,
                      masked, train_mask,
                      ridge_lambda, ridge_alpha, ridge_c, mu_alpha, mu_c)

    eigvals = jnp.linalg.eigvalsh(hessian)
    return hessian, eigvals


# init_hyperparams is an array
def optax_fit_RF_params(init_hyperparams, model, inputs, outputs, rewards,
                     masked=False, train_mask=None, max_iters=10_000, tol=1e-4,
                     ridge_lambda=0., ridge_alpha=0., ridge_c=0., mu_alpha=0., mu_c=0.,
                     explode_tol=1e7, burn_in_iters=1_000):

    meta_loss_fn = partial(compute_RF_objective, model=model, inputs=inputs, outputs=outputs, rewards=rewards,
                           masked=masked, train_mask=train_mask, ridge_lambda=ridge_lambda,
                           ridge_alpha=ridge_alpha, ridge_c=ridge_c, mu_alpha=mu_alpha, mu_c=mu_c)
    opt = optax.lbfgs(memory_size=50)
    value_and_grad_fun = optax.value_and_grad_from_state(meta_loss_fn)

    # Jit the step function for faster optimization iterations
    @jit
    def step(carry):
        hyperparams, state = carry
        value, grad = value_and_grad_fun(hyperparams, state=state)
        updates, state = opt.update(
            grad, state, hyperparams, value=value, grad=grad, value_fn=meta_loss_fn
        )
        hyperparams = optax.apply_updates(hyperparams, updates)
        return hyperparams, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = _tree_l2_norm(grad)

        finite = jnp.isfinite(err)

        within_burn_in = iter_num < burn_in_iters
        not_exploded = within_burn_in | (err < explode_tol)

        return (iter_num == 0) | ((iter_num < max_iters) & (err >= tol) & finite & not_exploded)

    carry = (init_hyperparams, opt.init(init_hyperparams))
    while bool(continuing_criterion(carry)):
        carry = step(carry)
    final_hyperparams, final_state = carry

    return final_hyperparams, final_state


def run_RF_epsilon_ball_test(hyperparams_arr, model, inputs, outputs, rewards, key,
                          masked=False, train_mask=None, num_samples=25, epsilon=1e-3,
                          ridge_lambda=0., ridge_alpha=0., ridge_c=0., mu_alpha=0., mu_c=0.):
    """
    Test whether hyperparams_arr is a local optimum by evaluating the objective at
    nearby points. Uses vmap to efficiently compute objectives at all perturbed
    points simultaneously.

    Note that `epsilon` is a single radius applied to every coordinate, even though
    the parameters live on different natural scales here (log-alphas, intercepts, and
    initial weights), so the perturbation is not equally meaningful in each direction.
    """
    result = 'FAILED'
    N = len(hyperparams_arr)
    curr_objective = compute_RF_objective(hyperparams_arr, model, inputs, outputs, rewards,
                                       masked, train_mask,
                                       ridge_lambda, ridge_alpha, ridge_c, mu_alpha, mu_c)

    # Generate random perturbations (num_samples, N)
    perturbations = jr.normal(key, (num_samples, N))
    perturbations /= jnp.linalg.norm(perturbations, axis=1, keepdims=True)
    perturbations *= epsilon
    perturbed_hyperparams = hyperparams_arr[jnp.newaxis, :] + perturbations  # (num_samples, N)

    # Vectorize objective computation over perturbed hyperparams
    def compute_objective_single(curr_hyperparams):
        return compute_RF_objective(curr_hyperparams, model, inputs, outputs, rewards,
                                 masked, train_mask,
                                 ridge_lambda, ridge_alpha, ridge_c, mu_alpha, mu_c)

    nearby_objectives = vmap(compute_objective_single)(perturbed_hyperparams)  # (num_samples,)

    if curr_objective < jnp.min(nearby_objectives):
        result = 'PASSED'

    return curr_objective, nearby_objectives, result


def run_RF_randomized_optimization_procedure(inputs, outputs, rewards, key, model,
                                          masked=False, train_mask=None, num_seeds=25, max_iters=500, tol=1e-4,
                                          alpha_init_scale=1e-2, c_init_scale=1e-1, w0_init_scale=1e-1,
                                          alpha_init_guess=None, c_init_guess=None, w0_init_guess=None,
                                          noise_level=0.01, ridge_lambda=0.,
                                          ridge_alpha=0., ridge_c=0., mu_alpha=0., mu_c=0.,
                                          explode_tol=1e7, burn_in_iters=1_000, 
                                          print_grad_and_hessian=False, run_epsilon_ball_test=False):

    num_alphas, num_baselines = REINFORCE_MODELS[model]

    all_hyperparams = []
    objective_vals = []
    bad_run_idxs = []

    print(f'Fitting model {model}: {num_alphas} learning rate(s), {num_baselines} intercept(s), '
          f'{NUM_WEIGHTS} initial weights ({get_num_params(model)} params total)', flush=True)
    print(f'Running {num_seeds} optimization seeds, with max_iters={max_iters}. Using ridge_lambda={ridge_lambda}, ridge_alpha={ridge_alpha}, ridge_c={ridge_c}.', flush=True)
    print(f'Using mu_alpha={mu_alpha}, mu_c={mu_c}.', flush=True)
    print(f'Using explode_tol={explode_tol}, burn_in_iters={burn_in_iters}.', flush=True)

    min_objective_val = None
    best_w0 = None
    best_hyperparams_arr = None
    for i in range(num_seeds):
        _, key, init_key, test_key = jr.split(key, 4)

        init_hyperparams = generate_RF_init_hyperparams(model, alpha_init_scale, c_init_scale, w0_init_scale, init_key,
                                                     alpha_init_guess=alpha_init_guess, c_init_guess=c_init_guess,
                                                     w0_init_guess=w0_init_guess, noise_level=noise_level)

        hyperparams_arr, final_state = optax_fit_RF_params(init_hyperparams, model, inputs, outputs, rewards,
                                                        masked=masked, train_mask=train_mask,
                                                        max_iters=max_iters, tol=tol, ridge_lambda=ridge_lambda,
                                                        ridge_alpha=ridge_alpha, ridge_c=ridge_c,
                                                        mu_alpha=mu_alpha, mu_c=mu_c,
                                                        explode_tol=explode_tol, burn_in_iters=burn_in_iters)
        all_hyperparams.append(hyperparams_arr)

        iter_count = int(otu.tree_get(final_state, 'count'))
        grad = otu.tree_get(final_state, 'grad')
        grad_norm = _tree_l2_norm(grad)

        converged = (grad_norm <= tol)

        inferred_objective_val = compute_RF_objective(hyperparams_arr, model, inputs, outputs, rewards,
                                                   masked, train_mask,
                                                   ridge_lambda, ridge_alpha, ridge_c, mu_alpha, mu_c)
        objective_vals.append(inferred_objective_val)

        # Print results from current optimization seed
        print(f'Iter {i+1} Inferred Objective: {inferred_objective_val}', flush=True)
        print(f'Iterations: {iter_count}/{max_iters}', flush=True)
        print(f'Converged: {converged}', flush=True)

        # run epsilon-ball test at candidate local optimum
        if run_epsilon_ball_test:
            _, _, result = run_RF_epsilon_ball_test(hyperparams_arr, model, inputs, outputs, rewards, test_key,
                                                masked=masked, train_mask=train_mask,
                                                ridge_lambda=ridge_lambda, ridge_alpha=ridge_alpha,
                                                ridge_c=ridge_c, mu_alpha=mu_alpha, mu_c=mu_c)
            print(f'Epsilon-ball test result: {result}', flush=True)

        # compute gradient and Hessian eigenvalues at candidate local optimum
        eigvals = None
        grad = None
        if print_grad_and_hessian and not jnp.isnan(inferred_objective_val):
            _, eigvals = compute_RF_hessian(hyperparams_arr, model, inputs, outputs, rewards,
                                         masked=masked, train_mask=train_mask, ridge_lambda=ridge_lambda,
                                         ridge_alpha=ridge_alpha, ridge_c=ridge_c, mu_alpha=mu_alpha, mu_c=mu_c)

            grad = jax.grad(compute_RF_objective)(hyperparams_arr, model, inputs, outputs, rewards,
                                               masked, train_mask,
                                               ridge_lambda, ridge_alpha, ridge_c, mu_alpha, mu_c)

            print(f'Grad: {grad}')
            print(f'Min/max eigvals: {eigvals.min(), eigvals.max()}', flush=True)

        print('\n', flush=True)

        curr_alphas = jnp.exp(hyperparams_arr[:num_alphas])
        curr_intercepts = hyperparams_arr[num_alphas:num_alphas + num_baselines]

        # slice alphas to match: the no-intercept models have curr_intercepts of shape (0,),
        # which will not broadcast against the (3,) alphas of the per-weight models.
        curr_baselines = curr_intercepts / curr_alphas[:num_baselines]

        print(f'Inferred alpha: {curr_alphas}, Inferred baselines: {curr_baselines}', flush=True)
        print(f'Inferred w_0: {hyperparams_arr[-NUM_WEIGHTS:]}')
        print('\n')

        # update min objective if you find a well-defined, lower objective than the current best AND if the optimizer converged
        if ((min_objective_val is None and not jnp.isnan(inferred_objective_val)) or (not jnp.isnan(inferred_objective_val) and inferred_objective_val < min_objective_val)) and converged:
            min_objective_val = inferred_objective_val
            best_w0 = hyperparams_arr[-NUM_WEIGHTS:]
            best_hyperparams_arr = hyperparams_arr

        if not converged:
            bad_run_idxs.append(i)

    all_hyperparams = jnp.asarray(all_hyperparams)
    objective_vals = jnp.asarray(objective_vals)

    print(f'Min Objective Value: {min_objective_val}', flush=True)

    if len(bad_run_idxs) > 0:
        bad_run_idxs = jnp.asarray(bad_run_idxs)
        bad_objectives = objective_vals[bad_run_idxs]
        print(f'min BAD objective value: {jnp.min(bad_objectives)}', flush=True)

        if min_objective_val is None or (jnp.min(bad_objectives) < min_objective_val):
            print('OPTIMIZATION ERROR. BAD SMALLER OBJECTIVE VALUE FOUND.', flush=True)
    else:
        print('SUCCESS. NO BAD OBJECTIVE VALUES FOUND.', flush=True)

    return best_w0, min_objective_val, best_hyperparams_arr, {'hyperparams': all_hyperparams, 'nlls': objective_vals, 'bad_run_idxs': bad_run_idxs}


def plot_weight_trajectory(true_hyp, inf_hyp, model, inputs, outputs, rewards):
    _, ws_inf, _ = run_single_RF_learning_forward_trajectory(inf_hyp, model, inputs, outputs, rewards)
    _, ws_true, _ = run_single_RF_learning_forward_trajectory(true_hyp, model, inputs, outputs, rewards)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        ax[i].plot(ws_true[:, i], color='black', linestyle='dashed')
        ax[i].plot(ws_inf[:, i], color='green')
    plt.show()


# CLUSTER FITTING ENTRY POINT
# ---------------------------------------------------------------------------------------------- #

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'witten_mouse_data.pkl')


def list_mouse_ids(data_path=DEFAULT_DATA_PATH):
    """Sorted mouse ids in the dataset.

    The ids are NON-CONTIGUOUS (13..43, then 901..914), so a SLURM array index must be
    mapped through this list -- using the index directly as a mouse_id would silently
    skip most of the animals.
    """
    with open(data_path, 'rb') as f:
        return sorted(pkl.load(f).keys())


def load_mouse_arrays(mouse_id, data_path=DEFAULT_DATA_PATH):
    """One mouse's (inputs, outputs, rewards), in the coding this pipeline expects.

    THE CHOICE FLIP: the dataset codes 0 = right / 1 = left, but `RF_log_likelihood`
    treats 1 as a rightward choice, so the choices are flipped here -- the same flip
    witten_psytrack_fitting applies. Without it the fitted right/left weights come out
    sign-mirrored. `inputs` is already [bias, right contrast, left contrast], which
    matches w = (bias, right, left), and rewards are unaffected by the flip.
    """
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
    if mouse_id not in data:
        raise SystemExit(f'mouse {mouse_id} not in {data_path}; available: {sorted(data.keys())}')
    mouse = data[mouse_id]

    inputs = jnp.asarray(np.asarray(mouse['inputs'], dtype=float))
    outputs = jnp.asarray(flip_outputs(np.asarray(mouse['outputs']).astype(int)))
    rewards = jnp.asarray(np.asarray(mouse['rewards'], dtype=float))

    assert inputs.ndim == 2 and inputs.shape[1] == NUM_WEIGHTS, f'expected (T,3) inputs, got {inputs.shape}'
    assert outputs.shape == (inputs.shape[0],), 'inputs/outputs length mismatch'
    return inputs, outputs, rewards


def main():
    p = argparse.ArgumentParser(
        description='Fit one REINFORCE model to one Witten mouse (one SLURM array task).',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # not `required`, so --list_mice works on its own; checked after that early exit
    p.add_argument('--model', choices=sorted(REINFORCE_MODELS))
    p.add_argument('--mouse_id', type=int, default=None, help='mouse id (omit if using --task_id)')
    p.add_argument('--task_id', type=int, default=None,
                   help='SLURM array index into the SORTED mouse-id list (not a mouse_id)')
    p.add_argument('--list_mice', action='store_true',
                   help='print the mouse ids and the array size, then exit')
    p.add_argument('--data_path', default=DEFAULT_DATA_PATH)

    # optimization
    p.add_argument('--num_seeds', type=int, default=200, help='random restarts')
    p.add_argument('--max_iters', type=int, default=2000, help='max L-BFGS iterations per restart')
    p.add_argument('--tol', type=float, default=1e-3, help='gradient-norm convergence tolerance')
    p.add_argument('--fit_seed', type=int, default=0, help='PRNG seed for the restarts')
    p.add_argument('--alpha_init_scale', type=float, default=1e0)
    # c = alpha*beta is O(alpha), so an init scale of order 1 badly inflates the fitted
    # intercept; 1e-4 keeps every restart near the plain-REINFORCE point c = 0.
    p.add_argument('--c_init_scale', type=float, default=1e-4)
    p.add_argument('--w0_init_scale', type=float, default=1e0)
    p.add_argument('--explode_tol', type=float, default=1e7)
    p.add_argument('--burn_in_iters', type=int, default=1_000)
    p.add_argument('--print_diagnostics', action='store_true', help='print epsilon ball test results, grad, hessian')

    # regularization -- off by default, i.e. a pure MLE fit
    p.add_argument('--ridge_lambda', type=float, default=0., help='ridge on w0')
    p.add_argument('--ridge_alpha', type=float, default=0., help='ridge on log-alpha')
    p.add_argument('--ridge_c', type=float, default=0., help='ridge on the intercept c')
    p.add_argument('--mu_alpha', type=float, default=0., help='prior mean for log-alpha')
    p.add_argument('--mu_c', type=float, default=0., help='prior mean for c')

    # output
    p.add_argument('--out_dir', default='reinforce_results', help='root directory for results')
    p.add_argument('--overwrite', action='store_true', help='refit even if the result exists')
    p.add_argument('--dry_run', action='store_true', help='tiny budget for a fast smoke test')
    args = p.parse_args()

    mouse_ids = list_mouse_ids(args.data_path)
    if args.list_mice:
        print(f'{len(mouse_ids)} mice (use --array=0-{len(mouse_ids) - 1}):')
        print('  ' + ', '.join(f'{i}->{m}' for i, m in enumerate(mouse_ids)))
        return
    if args.model is None:
        raise SystemExit('--model is required (or use --list_mice)')

    # resolve which mouse this invocation handles
    if args.task_id is not None:
        if not (0 <= args.task_id < len(mouse_ids)):
            raise SystemExit(f'--task_id {args.task_id} out of range 0..{len(mouse_ids) - 1}')
        mouse_id = mouse_ids[args.task_id]
    elif args.mouse_id is not None:
        mouse_id = args.mouse_id
    else:
        raise SystemExit('provide either --task_id or --mouse_id')

    if args.dry_run:
        args.num_seeds, args.max_iters = 2, 20

    out_dir = os.path.join(args.out_dir, args.model)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise SystemExit(f'ERROR: cannot create output directory {out_dir}: {e}')
    check_writable(out_dir)  # fail before fitting, not after a long run
    out_path = os.path.join(out_dir, f'mouse_{mouse_id}.pkl')
    if os.path.exists(out_path) and not args.overwrite:
        print(f'{out_path} already exists; use --overwrite to refit. Exiting.', flush=True)
        return

    inputs, outputs, rewards = load_mouse_arrays(mouse_id, args.data_path)
    T = int(outputs.shape[0])
    print(f'=== mouse {mouse_id} | model {args.model} | T={T} trials | '
          f'reward rate {float(jnp.mean(rewards)):.3f} ===', flush=True)

    t0 = time.time()
    best_w0, min_objective_val, best_hyperparams_arr, optim_dict = run_RF_randomized_optimization_procedure(
        inputs, outputs, rewards, jr.PRNGKey(args.fit_seed), args.model,
        num_seeds=args.num_seeds, max_iters=args.max_iters, tol=args.tol,
        alpha_init_scale=args.alpha_init_scale, c_init_scale=args.c_init_scale,
        w0_init_scale=args.w0_init_scale,
        ridge_lambda=args.ridge_lambda, ridge_alpha=args.ridge_alpha, ridge_c=args.ridge_c,
        mu_alpha=args.mu_alpha, mu_c=args.mu_c,
        explode_tol=args.explode_tol, burn_in_iters=args.burn_in_iters,
        run_epsilon_ball_test=args.print_diagnostics, print_grad_and_hessian=args.print_diagnostics,
    )
    elapsed = time.time() - t0

    # `run_RF_randomized_optimization_procedure` reports a best fit only among CONVERGED
    # seeds, returning None if none converged. Fall back to the best finite objective so a
    # long run is never discarded; `converged` records whether any seed met the tolerance.
    converged = best_hyperparams_arr is not None
    objective_vals = np.asarray(optim_dict['nlls'], dtype=float)
    if not np.any(np.isfinite(objective_vals)):
        raise SystemExit('ERROR: no seed produced a finite objective; nothing to save.')
    best_idx = int(np.nanargmin(np.where(np.isfinite(objective_vals), objective_vals, np.nan)))
    best_hyperparams_arr = optim_dict['hyperparams'][best_idx]
    min_objective_val = float(objective_vals[best_idx])
    best_w0 = best_hyperparams_arr[-NUM_WEIGHTS:]
    if not converged:
        print(f'WARNING: no seed converged; falling back to best finite seed '
              f'(idx {best_idx}, objective {min_objective_val:.4f})', flush=True)

    # Prior-free NLL on the full sequence: equals the objective only when unregularized,
    # and is what AIC must be built from.
    nll = float(run_single_RF_learning_forward_trajectory(
        best_hyperparams_arr, args.model, inputs, outputs, rewards)[0])
    num_params = get_num_params(args.model)
    aic = 2 * num_params + 2 * nll

    alphas, intercepts, w0 = unpack_RF_hyperparams(best_hyperparams_arr, args.model)
    num_alphas, num_intercepts = REINFORCE_MODELS[args.model]
    print(f'\nalpha={np.asarray(alphas)[:num_alphas]} '
          f'intercept={np.asarray(intercepts)[:num_intercepts]} w0={np.asarray(w0)}', flush=True)
    print(f'NLL={nll:.3f} ({nll / T:.5f}/trial, chance={np.log(2):.5f}) | '
          f'AIC={aic:.3f} ({num_params} params) | converged={converged} | '
          f'elapsed={elapsed:.1f}s', flush=True)

    record = dict(
        # identity
        mouse_id=mouse_id, model=args.model, num_params=num_params, num_trials=T,
        flipped_outputs=True, data_path=os.path.abspath(args.data_path),
        # headline results
        best_w0=best_w0, best_hyperparams_arr=best_hyperparams_arr,
        min_objective_val=min_objective_val, nll=nll, aic=aic,
        converged=converged, optim_dict=optim_dict,
        # fit settings
        num_seeds=args.num_seeds, max_iters=args.max_iters, tol=args.tol, fit_seed=args.fit_seed,
        alpha_init_scale=args.alpha_init_scale, c_init_scale=args.c_init_scale,
        w0_init_scale=args.w0_init_scale,
        ridge_lambda=args.ridge_lambda, ridge_alpha=args.ridge_alpha, ridge_c=args.ridge_c,
        mu_alpha=args.mu_alpha, mu_c=args.mu_c,
        elapsed_sec=elapsed,
    )
    record = to_numpy(record)  # drop JAX arrays -> loads anywhere, independent of JAX version

    # write to a temp file then rename, so a job killed mid-write (e.g. by a SLURM time
    # limit) cannot leave a half-written pickle for a later aggregation step to choke on
    tmp_path = f'{out_path}.tmp{os.getpid()}'
    with open(tmp_path, 'wb') as f:
        pkl.dump(record, f)
    os.replace(tmp_path, out_path)
    print(f'saved -> {out_path}', flush=True)


if __name__ == '__main__':
    main()


    