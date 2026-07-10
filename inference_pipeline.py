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

import matplotlib.pyplot as plt

from metric_GD_network import generate_randomized_multisession_graded_data, generate_metric_GD_train_test_split
from inference_utils import (apply_learning_updates, convert_weights_to_dict, generate_init_hyperparams,
                             get_depth, get_num_alphas, log_likelihood, model_forward,
                             squared_error_loss_fn, train_gd_network, unpack_hyperparams)
from utils import get_nonlinearity, calculate_running_avg_quantity, calculate_batched_quantity_per_session, generate_batched_randomized_multisession_graded_data, pad_sequence, make_train_mask, flatten_params
from globals import DEPTH_TO_NUM_WEIGHTS, MU_ALPHA, MU_BETA


@partial(jit, static_argnames=('nln','inference_model','alpha_W1'))
def run_train_test_single_learning_forward_trajectory(hyperparams, inputs, outputs, rewards, train_mask, nln, inference_model, alpha_W1):
    """Simulate the online-GD learning trajectory for a single sequence and split
    its negative log-likelihood into train and test portions via `train_mask`.

    Arguments:
    - hyperparams: 1-D array packed as (log_alphas, log_beta, w0)
    - inputs
    - outputs
    - rewards
    - train_mask: boolean array of shape (seq_len,); True = train, False = test
    - nln: non-linearity function applied to hidden state of a two-layer network
    - inference_model
    - alpha_W1 (only used if inference_model is 'fixed_metric_gd')

    Returns:
    - train_nll: negative log-likelihood summed over training time steps
    - test_nll:  negative log-likelihood summed over held-out time steps
    - ws:        weight trajectory (pytree, leading axis = time)
    - lls:       per-trial log-likelihoods (before masking)
    """

    # pass fixed_alpha_W1 by name (matching inference_utils.unpack_hyperparams signature)
    log_lrs, log_beta, w0 = unpack_hyperparams(hyperparams, inference_model, fixed_alpha_W1=alpha_W1)

    def _step(w, data_point):
        x, y, rew = data_point

        # compute GD gradients 
        _, gd_grads = value_and_grad(squared_error_loss_fn)(w, x, y, rew, nln)

        # now, apply diagonal/metric GD updates
        w_new = apply_learning_updates(w, gd_grads, log_lrs)

        # for the current weights compute the log-likelihood of the data given the hyperparams (alpha, beta)
        out = model_forward(w, x, nln)
        ll = log_likelihood(out, y, log_beta)

        return w_new, (ll, w_new)

    data_seq = (inputs, outputs, rewards)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    # total_nll = -jnp.sum(lls)
    train_nll = -jnp.sum(jnp.where(train_mask,  lls, 0.0))
    test_nll  = -jnp.sum(jnp.where(~train_mask, lls, 0.0))

    return train_nll, test_nll, ws, lls


@partial(jit, static_argnames=('nln','inference_model','alpha_W1'))
def run_single_learning_forward_trajectory(hyperparams, inputs, outputs, rewards, nln, inference_model, alpha_W1):
    """Simulate the online-GD learning trajectory for a single sequence and return
    its total negative log-likelihood over all trials.

    Arguments:
    - hyperparams: 1-D array packed as (log_alphas, log_beta, w0)
    - inputs
    - outputs
    - rewards
    - nln: non-linearity function applied to hidden state of a two-layer network
    - inference_model
    - alpha_W1 (only used if inference_model is 'fixed_metric_gd')

    Returns:
    - total_nll: negative log-likelihood summed over all time steps
    - ws:        weight trajectory (pytree, leading axis = time)
    - lls:       per-trial log-likelihoods
    """

    # # pass fixed_alpha_W1 by name (matching inference_utils.unpack_hyperparams signature)
    # log_lrs, log_beta, w0 = unpack_hyperparams(hyperparams, inference_model, fixed_alpha_W1=alpha_W1)

    # def _step(w, data_point):
    #     x, y, rew = data_point

    #     # compute GD gradients 
    #     _, gd_grads = value_and_grad(squared_error_loss_fn)(w, x, y, rew, nln)

    #     # now, apply diagonal/metric GD updates
    #     w_new = apply_learning_updates(w, gd_grads, log_lrs)

    #     # for the current weights compute the log-likelihood of the data given the hyperparams (alpha, beta)
    #     out = model_forward(w, x, nln)
    #     ll = log_likelihood(out, y, log_beta)

    #     return w_new, (ll, w_new)

    # data_seq = (inputs, outputs, rewards)
    # _, (lls, ws) = lax.scan(_step, w0, data_seq)

    # total_nll = -jnp.sum(lls)
    total_nll, _, ws, lls = run_train_test_single_learning_forward_trajectory(hyperparams, inputs, outputs, rewards, jnp.ones_like(outputs, dtype=bool), nln=nln, inference_model=inference_model, alpha_W1=alpha_W1)

    return total_nll, ws, lls

# following functions operate on batches of (inputs, outputs, rewards), and return a batched array of total nlls and batched pytree of weight trajectories
# batched_run_forward_learning_trajectory = jit(vmap(run_single_learning_forward_trajectory, in_axes=(None, 0, 0, 0, None, None, None)), static_argnames=('nln','inference_model','alpha_W1'))


@partial(jit, static_argnames=('nln','inference_model','alpha_W1'))
def batched_run_forward_learning_trajectory(hyperparams, inputs, outputs, rewards, lengths, nln, inference_model, alpha_W1):
    """Simulate the online-GD learning trajectory for a padded batch of sequences
    (vmapped over the batch), ignoring padded time steps via `lengths`.

    Arguments:
    - hyperparams: 1-D array packed as (log_alphas, log_beta, w0)
    - inputs: array of shape (batch_size, max_seq_len, input_dim)
    - outputs: array of shape (batch_size, max_seq_len) containing integer class labels
    - rewards: array of shape (batch_size, max_seq_len) containing rewards for each trial
    - lengths: array of shape (batch_size,) indicating the length of each sequence in the batch
    - nln: non-linearity function applied to hidden state of a two-layer network
    - inference_model
    - alpha_W1 (only used if inference_model is 'fixed_metric_gd')

    Returns:
    - total_nlls: (batch_size,) negative log-likelihood per sequence (padding excluded)
    - all_ws:     batched weight trajectories (pytree, axes = (batch, time, ...))
    - all_lls:    (batch_size, max_seq_len) per-trial log-likelihoods (0 on padding)
    """

    # pass fixed_alpha_W1 by name (matching inference_utils.unpack_hyperparams signature)
    T_max = inputs.shape[1]
    length_mask = jnp.arange(T_max)[jnp.newaxis, :] < lengths[:, jnp.newaxis]  # (batch_size, max_seq_len)
    log_lrs, log_beta, w0 = unpack_hyperparams(hyperparams, inference_model, fixed_alpha_W1=alpha_W1)

    def _step(w, data_point):
        x, y, rew, mask_val = data_point
        
        # prevent NaNs from permeating the backward gradient computation later during inference
        safe_x = jnp.where(mask_val, x, jnp.zeros_like(x))
        safe_y = jnp.where(mask_val, y, jnp.zeros_like(y))
        safe_rew = jnp.where(mask_val, rew, jnp.zeros_like(rew))

        # compute GD gradients only if mask_val is 1 (valid time step), otherwise return zero gradients to avoid updating weights based on padded time steps
        # gd_grads = jnp.where(mask_val == 1, grad(squared_error_loss_fn)(w, x, y, rew, nln), jnp.zeros_like(w))
        raw_grads = grad(squared_error_loss_fn)(w, safe_x, safe_y, safe_rew, nln)
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, raw_grads)
        gd_grads = jax.tree_util.tree_map(
            lambda g, z: jnp.where(mask_val, g, z),
            raw_grads, zero_grads
        )   

        # now, apply GD update
        w_candidate = apply_learning_updates(w, gd_grads, log_lrs)

        w_new = jax.tree_util.tree_map(
            lambda wn, wo: jnp.where(mask_val, wn, wo),
            w_candidate, w
        )

        # for the current weights compute the log-likelihood of the data given the hyperparams (alpha, beta)
        out = model_forward(w, safe_x, nln)

        # mask_val is 1 for valid time steps and 0 for padding, so we multiply the 
        # log-likelihood by the mask to zero out contributions from padded time steps
        ll = jnp.where(mask_val, log_likelihood(out, safe_y, log_beta), 0.)

        return w_new, (ll, w_new)
    
    def scan_single_sequence(seq_inputs, seq_outputs, seq_rewards, seq_mask):
        seq_data = (seq_inputs, seq_outputs, seq_rewards, seq_mask)
        _, (lls, ws) = lax.scan(_step, w0, seq_data)
        total_nll = -jnp.sum(lls)
        return total_nll, ws, lls 

    # vmap over batch dimension to compute nlls and weight trajectories for each sequence in the batch
    total_nlls, all_ws, all_lls = vmap(scan_single_sequence)(inputs, outputs, rewards, length_mask)

    return total_nlls, all_ws, all_lls


@partial(jit, static_argnames=('nln','inference_model','batched', 'masked', 'alpha_W1', 'num_weights', 'ridge_lambda', 'alphabeta_ridge', 'mu_alpha', 'mu_beta'))
def compute_nll(hyperparams, inputs, outputs, rewards, nln, inference_model, batched,
                lengths=None, masked=False, train_mask=None, alpha_W1=None, num_weights=9,
                ridge_lambda=0., alphabeta_ridge=0., mu_alpha=0., mu_beta=0.):
    N = len(hyperparams)
    nll = None
    # `masked` is a static (compile-time) flag, so we pick the branch at trace time
    # with a plain Python if/elif/else. This avoids lax.cond tracing the train-mask
    # branch when train_mask is None (the plain single-animal case).
    if batched:
        nlls, _, _ = batched_run_forward_learning_trajectory(hyperparams, inputs, outputs, rewards, lengths, nln, inference_model, alpha_W1)
        nll = jnp.mean(nlls)
    elif masked:
        # train/test split: score only the training trials (train_mask: True = train)
        nll = run_train_test_single_learning_forward_trajectory(hyperparams, inputs, outputs, rewards, train_mask, nln, inference_model, alpha_W1)[0]
    else:
        nll = run_single_learning_forward_trajectory(hyperparams, inputs, outputs, rewards, nln, inference_model, alpha_W1)[0]

    l2_prior = ridge_lambda * jnp.sum(hyperparams[-num_weights:]**2) + alphabeta_ridge * jnp.sum((hyperparams[:N-num_weights-1] - mu_alpha)**2) + alphabeta_ridge * (hyperparams[N-num_weights-1] - mu_beta)**2
    nll += l2_prior

    return nll


@partial(jit, static_argnames=('nln','inference_model','batched', 'masked', 'alpha_W1', 'num_weights', 'ridge_lambda', 'alphabeta_ridge', 'mu_alpha', 'mu_beta'))
def compute_hessian(hyperparams, inputs, outputs, rewards, nln, inference_model, batched,
                    lengths=None, masked=False, train_mask=None, alpha_W1=None, num_weights=9, 
                    ridge_lambda=0., alphabeta_ridge=0., mu_alpha=0., mu_beta=0.):
    hess_fn = jax.hessian(compute_nll)
    hessian = hess_fn(hyperparams, inputs, outputs, rewards, nln, inference_model, batched, 
                      lengths, masked, train_mask, alpha_W1, num_weights, 
                      ridge_lambda, alphabeta_ridge, mu_alpha, mu_beta)
    
    eigvals = jnp.linalg.eigvalsh(hessian)
    return hessian, eigvals 


# init_hyperparams is an array
def optax_fit_params(init_hyperparams, inputs, outputs, rewards, nln, inference_model, 
                     batched, lengths=None, masked=False, train_mask=None,
                     alpha_W1=None, max_iters=10_000, tol=1e-4, num_weights=9, ridge_lambda=0.,
                     alphabeta_ridge=0., mu_alpha=0., mu_beta=0., explode_tol=1e7, burn_in_iters=1_000):

    meta_loss_fn = partial(compute_nll, inputs=inputs, outputs=outputs, rewards=rewards, nln=nln, 
                           inference_model=inference_model, batched=batched, lengths=lengths, masked=masked, train_mask=train_mask,
                           alpha_W1=alpha_W1, num_weights=num_weights, ridge_lambda=ridge_lambda, alphabeta_ridge=alphabeta_ridge,
                           mu_alpha=mu_alpha, mu_beta=mu_beta)
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
        err = otu.tree_norm(grad)

        finite = jnp.isfinite(err)
        
        within_burn_in = iter_num < burn_in_iters
        not_exploded = within_burn_in | (err < explode_tol)

        return (iter_num == 0) | ((iter_num < max_iters) & (err >= tol) & finite & not_exploded)

    carry = (init_hyperparams, opt.init(init_hyperparams))
    while bool(continuing_criterion(carry)):
        carry = step(carry)
    final_hyperparams, final_state = carry

    return final_hyperparams, final_state


def run_epsilon_ball_test(hyperparams_arr, inputs, outputs, rewards, key, nln, inference_model, batched, 
                          lengths=None, masked=False, train_mask=None,
                          alpha_W1=None, num_samples=25, epsilon=0.01, num_weights=9, ridge_lambda=0.,
                          alphabeta_ridge=0., mu_alpha=0., mu_beta=0.):
    """
    Test whether hyperparams_arr is a local optimum by evaluating NLL at nearby points.
    Uses vmap to efficiently compute NLLs at all perturbed points simultaneously.
    """
    result = 'FAILED'
    N = len(hyperparams_arr)
    curr_nll = compute_nll(hyperparams_arr, inputs, outputs, rewards, nln, inference_model, batched, lengths, 
                           masked, train_mask, alpha_W1, num_weights, ridge_lambda, alphabeta_ridge, mu_alpha, mu_beta)

    
    # Generate random perturbations (num_samples, N)
    perturbations = jr.normal(key, (num_samples, N)) 
    perturbations /= jnp.linalg.norm(perturbations, axis=1, keepdims=True)
    perturbations *= epsilon 
    perturbed_hyperparams = hyperparams_arr[jnp.newaxis, :] + perturbations  # (num_samples, N)
    
    # Vectorize NLL computation over perturbed hyperparams
    def compute_nll_single(curr_hyperparams):
        return compute_nll(curr_hyperparams, inputs, outputs, rewards, nln, inference_model, batched, lengths,
                           masked, train_mask, alpha_W1, num_weights, ridge_lambda, alphabeta_ridge, mu_alpha, mu_beta)
    
    nearby_nlls = vmap(compute_nll_single)(perturbed_hyperparams)  # (num_samples,)
    
    if curr_nll < jnp.min(nearby_nlls):
        result = 'PASSED'

    return curr_nll, nearby_nlls, result


def run_randomized_optimization_procedure(inputs, outputs, rewards, key, depth, nln, inference_model, batched, lengths=None,
                                          masked=False, train_mask=None, alpha_W1=None, num_seeds=25, max_iters=500, tol=1e-4,
                                          alpha_init_scale=1e-1, beta_init_scale=1e0, w0_init_scale=1e-1,
                                          alpha_init_guess=None, beta_init_guess=None, w0_init_guess=None,
                                          noise_level=0.01, epsilon_seeds=25, epsilon=0.01, ridge_lambda=0.,
                                          alphabeta_ridge=0., mu_alpha=0., mu_beta=0., 
                                          explode_tol=1e7, burn_in_iters=1_000):
    
    num_alphas = get_num_alphas(inference_model, depth)

    all_hyperparams = []
    objective_vals = []
    bad_run_idxs = []

    num_weights = DEPTH_TO_NUM_WEIGHTS[depth]
    print(f'num weights in model: {num_weights}')

    min_objective_val = None 
    best_w0 = None 
    best_hyperparams_arr = None
    for i in range(num_seeds):
        _, key, init_key, test_key = jr.split(key, 4)

        init_hyperparams = generate_init_hyperparams(inference_model, depth, alpha_init_scale, beta_init_scale, w0_init_scale, init_key,
                                                     alpha_init_guess=alpha_init_guess, beta_init_guess=beta_init_guess, w0_init_guess=w0_init_guess, 
                                                     noise_level=noise_level)

        hyperparams_arr, final_state = optax_fit_params(init_hyperparams, inputs, outputs, rewards, nln, inference_model, 
                                                        batched, lengths=lengths, masked=masked, train_mask=train_mask, alpha_W1=alpha_W1, 
                                                        max_iters=max_iters, tol=tol, num_weights=num_weights, ridge_lambda=ridge_lambda,
                                                        alphabeta_ridge=alphabeta_ridge, mu_alpha=mu_alpha, mu_beta=mu_beta, 
                                                        explode_tol=explode_tol, burn_in_iters=burn_in_iters)
        all_hyperparams.append(hyperparams_arr)

        iter_count = int(otu.tree_get(final_state, 'count'))
        grad = otu.tree_get(final_state, 'grad')
        grad_norm = otu.tree_norm(grad)

        converged = (grad_norm <= tol)

        inferred_objective_val = compute_nll(hyperparams_arr, inputs, outputs, rewards, nln, inference_model, batched, lengths, 
                                             masked, train_mask, alpha_W1, num_weights, ridge_lambda, alphabeta_ridge, mu_alpha, mu_beta)
        objective_vals.append(inferred_objective_val)

        # run epsilon-ball test at candidate local optimum
        _, _, result = run_epsilon_ball_test(hyperparams_arr, inputs, outputs, rewards, test_key, nln, inference_model, batched, lengths=lengths,
                                             masked=masked, train_mask=train_mask, alpha_W1=alpha_W1, num_samples=epsilon_seeds, epsilon=epsilon, 
                                             num_weights=num_weights, ridge_lambda=ridge_lambda, alphabeta_ridge=alphabeta_ridge, 
                                             mu_alpha=mu_alpha, mu_beta=mu_beta)

        # compute gradient and Hessian eigenvalues at candidate local optimum
        eigvals = None 
        grad = None
        if not jnp.isnan(inferred_objective_val):
            _, eigvals = compute_hessian(hyperparams_arr, inputs, outputs, rewards, nln, inference_model, batched, lengths=lengths, 
                                         masked=masked, train_mask=train_mask,
                                         alpha_W1=alpha_W1, num_weights=num_weights, ridge_lambda=ridge_lambda,
                                         alphabeta_ridge=alphabeta_ridge, mu_alpha=mu_alpha, mu_beta=mu_beta)

            grad = jax.grad(compute_nll)(hyperparams_arr, inputs, outputs, rewards, nln, inference_model, batched, lengths=lengths, 
                                         masked=masked, train_mask=train_mask,
                                         alpha_W1=alpha_W1, num_weights=num_weights, ridge_lambda=ridge_lambda,
                                         alphabeta_ridge=alphabeta_ridge, mu_alpha=mu_alpha, mu_beta=mu_beta)
            
            # kwargs = dict(inputs=inputs, outputs=outputs, rewards=rewards, nln=nln,
            #   inference_model=inference_model, batched=batched,
            #   alpha_W1=alpha_W1, num_weights=num_weights, ridge_lambda=ridge_lambda,
            #   alphabeta_ridge=alphabeta_ridge, mu_alpha=mu_alpha, mu_beta=mu_beta)
            # g_fd = finite_diff_gradient(hyperparams_arr, 1e-6, **kwargs)
            # g_fd_norm = float(jnp.linalg.norm(g_fd))
            # relative_g = g_fd_norm / (abs(float(inferred_nll)) + 1e-8)
            # print(f'Norm check. FD grad: {g_fd}, Relative grad strength: {relative_g}')

        # Print results from current optimization seed
        print(f'Iter {i+1} Inferred Objective Value: {inferred_objective_val} \nGradient: {grad} \nHessian eigvals: {eigvals}')
        print(f'Iterations: {iter_count}/{max_iters}')
        print(f'Converged: {converged}')
        print(f'Epsilon-ball test result: {result}') # if desired, can also print f'Nearby NLLs: {nearby_nlls}'
        print('\n')

        curr_alphas = jnp.exp(hyperparams_arr[:num_alphas])
        curr_beta = jnp.exp(hyperparams_arr[num_alphas])

        print(f'Inferred alpha: {curr_alphas}, Inferred beta: {curr_beta}')
        if depth == 'two-layer':
            print(f'Inferred W1_diag_0: {hyperparams_arr[num_alphas+1:num_alphas+4]}, Inferred W2_0: {hyperparams_arr[num_alphas+4:].reshape((2, 3))}')
        else:
            print(f'Inferred W_0: {hyperparams_arr[num_alphas+1:].reshape((2, 3))}')
        print('\n\n')

        # update min_nll if you find a well-defined, lower nll than the current best AND if the optimizer converged
        if ((min_objective_val is None and not jnp.isnan(inferred_objective_val)) or (not jnp.isnan(inferred_objective_val) and inferred_objective_val < min_objective_val)) and converged:
            min_objective_val = inferred_objective_val
            best_w0 = convert_weights_to_dict(hyperparams_arr[num_alphas+1:])
            best_hyperparams_arr = hyperparams_arr
        
        if not converged:
            bad_run_idxs.append(i)
    
    all_hyperparams = jnp.asarray(all_hyperparams)
    objective_vals = jnp.asarray(objective_vals)

    print(f'Min Objective Value: {min_objective_val}')

    if len(bad_run_idxs) > 0:
        bad_run_idxs = jnp.asarray(bad_run_idxs)
        bad_nlls = objective_vals[bad_run_idxs]
        print(f'min BAD objective value: {jnp.min(bad_nlls)}')

        if jnp.min(bad_nlls) < min_objective_val:
            print('OPTIMIZATION ERROR. BAD SMALLER OBJECTIVE VALUE FOUND.')
    else:
        print('SUCCESS. NO BAD OBJECTIVE VALUES FOUND.')

    return best_w0, min_objective_val, best_hyperparams_arr, {'hyperparams': all_hyperparams, 'nlls' : objective_vals, 'bad_run_idxs': bad_run_idxs}


# for debugging purposes
def finite_diff_gradient(x0, eps=1e-5, **kwargs):
    n = len(x0)
    g_fd = np.zeros(n)
    f0 = float(compute_nll(x0, **kwargs))
    for i in range(n):
        e_i = jnp.zeros(n).at[i].set(1.0)
        fp = float(compute_nll(x0 + eps * e_i, **kwargs))
        fm = float(compute_nll(x0 - eps * e_i, **kwargs))
        g_fd[i] = (fp - fm) / (2 * eps)
    return jnp.array(g_fd)

def multi_scale_fd_gradient(x0, eps_list, **kwargs):
    results = {}
    n = len(x0)
    for eps in eps_list:
        g = finite_diff_gradient(x0, eps, **kwargs)
        results[eps] = (jnp.array(g), np.linalg.norm(g))
        print(f"eps={eps:.0e}: ||g_fd||={results[eps][1]:.4f},  g={g}")
    return results


def visualize_inferred_vs_true_effective_weights(inferred_hyperparams_arr, true_hyperparams_arr, inputs, outputs, rewards, 
                                                 nln, inference_model, alpha_W1=None):
    
    depth = get_depth(true_hyperparams_arr, inference_model)
    nrows = 3
    if depth == 'one-layer':
        nrows = 2
    
    true_nll, true_ws, _ = run_single_learning_forward_trajectory(true_hyperparams_arr, inputs, outputs, rewards, nln, inference_model, alpha_W1)

    inferred_nll, inferred_ws, _ = run_single_learning_forward_trajectory(inferred_hyperparams_arr, inputs, outputs, rewards, nln, inference_model, alpha_W1)

    fig, axs = plt.subplots(nrows, 3, figsize=(18, 5*nrows), sharex=True, sharey=True)

    true_trajectories = flatten_params(true_ws)
    inferred_trajectories = flatten_params(inferred_ws)

    j = 0
    for ax in axs.ravel():
        ax.plot(inferred_trajectories[:, j], lw=3., color='black', linestyle='dashed')
        ax.plot(true_trajectories[:, j], lw=3., color='green')
        ax.axhline(0, lw=2., color='blue', linestyle=':')
        j += 1

    plt.show()

    print(f'True Data NLL (no prior): {true_nll}')
    print(f'Inferred NLL (no prior): {inferred_nll}')

    return


def train_test_exploration():
    key = jr.PRNGKey(38)
    skeys = jr.split(key, 5)
    num_sessions = 40
    num_gradings = 4
    num_samples_per_grading = 30
    num_zero_contrast_per_session = 10
    input_dim = 3
    depth = 'two-layer'
    nln_name = 'erf'
    nln = get_nonlinearity(nln_name)
    inference_model = 'gd'
    alpha_W1 = None

     # generate initial hyperparams for ground truth network
    alpha = 0.005
    beta = 8.0
    w0_arr = jnp.array([0.5, 0.3, 0.1, 0.2, -0.1, 0.05, -0.3, 0.4, -0.2])   # for two-layer model
    w0 = convert_weights_to_dict(w0_arr)

    inputs, targets, mask = generate_randomized_multisession_graded_data(num_sessions, num_gradings, num_samples_per_grading, num_zero_contrast_per_session, skeys[0])

    _, _, results_dict = train_gd_network(nln_name, w0, inputs, targets, alpha, beta, skeys[1], num_grading_levels=num_gradings)
    outputs = jnp.array(results_dict['outputs'])    # this is a 1d array
    rewards = jnp.array(results_dict['rewards'])    # this is a 1d array
    train_mask = make_train_mask(inputs.shape[0], block_length=20, test_fraction=0.1)

    best_w0, _, best_hyperparams_arr, optim_dict = run_randomized_optimization_procedure(inputs, outputs, rewards, skeys[2], depth, nln, inference_model, batched=False,
                                                                                               masked=True, train_mask=train_mask,
                                                                                               alpha_W1=None, num_seeds=10, max_iters=1_000, tol=1e-4,
                                                                                               alpha_init_scale=1e-1, beta_init_scale=10e0, w0_init_scale=1e-1,
                                                                                               ridge_lambda=0.5, alphabeta_ridge=0.5, mu_alpha=MU_ALPHA, mu_beta=MU_BETA)
                                
    print(f'Inferred alpha: {jnp.exp(best_hyperparams_arr[0])}')
    print(f'Inferred beta: {jnp.exp(best_hyperparams_arr[1])}')
    print(f'Inferred w0: {best_w0}')

    true_hyperparams_arr = jnp.concatenate((jnp.log(jnp.array([alpha, beta])), w0_arr))

    true_train_nll, true_test_nll, _, _ = run_train_test_single_learning_forward_trajectory(true_hyperparams_arr, inputs, outputs, rewards, train_mask, nln, inference_model, alpha_W1=None)

    inferred_train_nll, inferred_test_nll, _, _ = run_train_test_single_learning_forward_trajectory(best_hyperparams_arr, inputs, outputs, rewards, train_mask, nln, inference_model, alpha_W1=None)

    print(f'True Train NLL (no prior): {true_train_nll}, True Test NLL (no prior): {true_test_nll}')
    print(f'Inferred Train NLL (no prior): {inferred_train_nll}, Inferred Test NLL (no prior): {inferred_test_nll}')


if __name__ == "__main__":
    train_test_exploration()
    # print(1./0)
    # test out batched inference procedure on a synthetic dataset and a ground truth network with known hyperparameters, and visualize the inferred vs true effective weight trajectories
    key = jr.PRNGKey(38)
    keys = jr.split(key, 5)
    skey = keys[0]
    sskey = keys[1]
    all_num_sessions = jnp.array([40, 30, 50, 35, 40, 45])
    num_train = 3
    num_gradings = 4
    num_samples_per_grading = 30
    num_zero_contrast_per_session = 10
    input_dim = 3
    depth = 'two-layer'
    nln_name = 'erf'
    nln = get_nonlinearity(nln_name)
    inference_model = 'gd'
    alpha_W1 = None

    # set different lengths to test out variable-length sequence handling in the batched inference procedure
    lengths =  (2 * num_gradings * num_samples_per_grading + num_zero_contrast_per_session) * all_num_sessions
    max_length = jnp.max(lengths)

    # generate initial hyperparams for ground truth network
    alpha = 0.005
    beta = 8.0
    w0_arr = jnp.array([0.5, 0.3, 0.1, 0.2, -0.1, 0.05, -0.3, 0.4, -0.2])   # for two-layer model
    w0 = convert_weights_to_dict(w0_arr)

    all_inputs = None
    all_outputs = None 
    all_rewards = None

    # for each dataset, get the sequence of outputs and rewards and amass them into a big batch for inference. Note that the sequences are of different lengths, so we will need to pad them to get a single array for inputs, outputs, and rewards, and also create a lengths array to indicate the valid time steps for each sequence in the batch
    for num_sessions in all_num_sessions:
        skey, _ = jr.split(skey)
        inputs, targets, mask = generate_randomized_multisession_graded_data(num_sessions, num_gradings, num_samples_per_grading, num_zero_contrast_per_session, key)

        _, _, results_dict = train_gd_network(nln_name, w0, inputs, targets, alpha, beta, skey, num_grading_levels=num_gradings)
        outputs = jnp.array(results_dict['outputs'])    # this is a 1d array
        rewards = jnp.array(results_dict['rewards'])    # this is a 1d array

        # pad sequences to max_length
        seq_length = outputs.shape[0]
        pad_length = max_length - seq_length
        padded_inputs = jnp.pad(inputs, ((0, pad_length), (0, 0)))   # pad only the time dimension
        padded_outputs = jnp.pad(outputs, (0, pad_length))    # pad the 1d outputs array
        padded_rewards = jnp.pad(rewards, (0, pad_length))    # pad the 1d rewards array

        if all_inputs is None:
            all_inputs = padded_inputs[jnp.newaxis, :]
            all_outputs = padded_outputs[jnp.newaxis, :]
            all_rewards = padded_rewards[jnp.newaxis, :]
        else:
            all_inputs = jnp.concatenate((all_inputs, padded_inputs[jnp.newaxis, :]), axis=0)
            all_outputs = jnp.concatenate((all_outputs, padded_outputs[jnp.newaxis, :]), axis=0)
            all_rewards = jnp.concatenate((all_rewards, padded_rewards[jnp.newaxis, :]), axis=0)
    
    print(f'all_inputs shape: {all_inputs.shape}, all_outputs shape: {all_outputs.shape}, all_rewards shape: {all_rewards.shape}')

    train_inputs = all_inputs[:num_train]
    train_outputs = all_outputs[:num_train]
    train_rewards = all_rewards[:num_train]
    train_lengths = lengths[:num_train]

    test_inputs = all_inputs[num_train:]
    test_outputs = all_outputs[num_train:]
    test_rewards = all_rewards[num_train:]
    test_lengths = lengths[num_train:]

    # run inference procedure to get inferred hyperparams
    best_w0, min_nll, best_hyperparams_arr, optim_dict = run_randomized_optimization_procedure(train_inputs, train_outputs, train_rewards, sskey, depth, nln, inference_model, batched=True, lengths=train_lengths,
                                                                    alpha_W1=None, num_seeds=2, max_iters=1_000, tol=1e-4,
                                                                    alpha_init_scale=1e-1, beta_init_scale=10e0, w0_init_scale=1e-1,
                                                                    ridge_lambda=0.5, alphabeta_ridge=0.5, mu_alpha=MU_ALPHA, mu_beta=MU_BETA)
    
    print(f'Inferred alpha: {jnp.exp(best_hyperparams_arr[0])}')
    print(f'Inferred beta: {jnp.exp(best_hyperparams_arr[1])}')
    print(f'Inferred w0: {best_w0}')

    true_hyperparams_arr = jnp.concatenate((jnp.log(jnp.array([alpha, beta])), w0_arr))
    true_nlls, _, _ = batched_run_forward_learning_trajectory(true_hyperparams_arr, train_inputs, train_outputs, train_rewards, train_lengths, nln, inference_model, alpha_W1=None)
    inferred_nlls, _, _ = batched_run_forward_learning_trajectory(best_hyperparams_arr, train_inputs, train_outputs, train_rewards, train_lengths, nln, inference_model, alpha_W1=None) 


    print(f'true_nlls on train data: {true_nlls}')
    print(f'inferred_nlls on train data: {inferred_nlls}')

    # nlls on test inputs
    true_nlls_test, _, _ = batched_run_forward_learning_trajectory(true_hyperparams_arr, test_inputs, test_outputs, test_rewards, test_lengths, nln, inference_model, alpha_W1=None)
    inferred_nlls_test, _, _ = batched_run_forward_learning_trajectory(best_hyperparams_arr, test_inputs, test_outputs, test_rewards, test_lengths, nln, inference_model, alpha_W1=None)

    print(f'true_nlls on test data: {true_nlls_test}')
    print(f'inferred_nlls on test data: {inferred_nlls_test}')

    # visualize inferred vs true effective weight trajectories on train data
    visualize_inferred_vs_true_effective_weights(best_hyperparams_arr, true_hyperparams_arr, test_inputs[0], test_outputs[0], test_rewards[0], 
                                                 nln, inference_model, alpha_W1=None)