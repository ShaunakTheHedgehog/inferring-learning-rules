import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax import debug
from jax import hessian
from jax.tree_util import tree_map
import optax
import optax.tree_utils as otu

from jaxopt import ScipyMinimize, ScipyBoundedMinimize, BFGS
from functools import partial
import numpy as np

import pickle as pkl

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pdb

# from flax import linen as nn
import functools
import random
import math
from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from utils import generate_multisession_graded_data, calculate_model_agnostic_bias, calculate_session_accs, flatten_params, turn_pytree_to_arr

def run_forward_weight_trajectory(hyperparams, w0, inputs, outputs, rewards, alpha_scale=1., hyperparams_is_list=False):
    """
    Arguments:
    - hyperparams (alpha, beta)
    - w0
    - inputs
    - outputs
    - rewards

    Returns:
    - total negative log-likelihood of the data given the hyperparams (alpha, beta)
    """
    if hyperparams_is_list:
        alpha = hyperparams[0]
        beta = hyperparams[1]
    else:
        alpha = hyperparams['alpha']       
        beta = hyperparams['beta']

    @jit
    def _log_likelihood(out, y):
        '''
        out : the current output of the model 
        y   : the output that was made by the agent (0 = R, 1 = L)
        '''
        Q_y = out[y]
        Q_opp = out[1-y]
        p_y_given_w_and_x = 1.0 / (1.0 + jnp.exp(-jnp.exp(beta) * (Q_y - Q_opp)))
        ll = jnp.log(p_y_given_w_and_x)
        return ll

    # Loss: squared error loss
    @jit
    def _squared_error_loss_fn(w, x, y_choice, rew):
        out, _ = cortico_striatal_net(w, x)
        Q_y = out[y_choice]
        return 0.5 * (Q_y - rew)**2
    
    @jit
    def _stim_loss(w, x, y_choice, rew):
        W1_diag = w["W1_diag"]     # (3,)
        W2 = w["W2"]               # (2, 3)

        Q_stim = jnp.dot(W2[y_choice, 1:], W1_diag[1:] * x[1:])
        loss = 0.5 * (rew - Q_stim)**2
        return loss

    @jit
    def _const_loss(w, x, y_choice, rew):
        W1_diag = w["W1_diag"]     # (3,)
        W2 = w["W2"]               # (2, 3)

        Q_const = W2[y_choice, 0] * W1_diag[0] * x[0]
        loss = 0.5 * (rew - Q_const)**2
        return loss
    
    @jit 
    def _tutor_executor_loss_fn(w, x, y_choice, rew):
        return _stim_loss(w, x, y_choice, rew) + _const_loss(w, x, y_choice, rew)

    def _step(w, data_point):
        x, y, rew = data_point

        # first, apply the standard single loss GD learning rule to get the next weights w
        _, gd_grads = value_and_grad(_squared_error_loss_fn)(w, x, y, rew)
        w_new = jax.tree_util.tree_map(lambda p, g: p - (jnp.exp(alpha) * alpha_scale) * g, w, gd_grads)

        # _, te_grads_W1 = jax.value_and_grad(_squared_error_loss_fn)(w, x, y, rew)
        # _, te_grads_W2 = jax.value_and_grad(_tutor_executor_loss_fn)(w, x, y, rew)

        # w_new["W1_diag"] -= (alpha2 * alpha_scale) * gd_grads["W1_diag"]
        # w_new["W2"] -= (alpha2 * alpha_scale) * te_grads_W2["W2"]

        # second, for the current weights compute the log-likelihood of the data given the hyperparams (alpha, beta)
        out, _ = cortico_striatal_net(w, x)
        ll = _log_likelihood(out, y)

        return w_new, (ll, w_new)

    data_seq = (inputs, outputs, rewards)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    total_nll = -jnp.sum(lls)

    return total_nll, ws 


def compute_nll(hyperparams, w0, inputs, outputs, rewards, alpha_scale=1., hyperparams_is_list=False):
    total_nll, _ = run_forward_weight_trajectory(hyperparams, w0, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=hyperparams_is_list)
    return total_nll

def compute_hessian(hyperparams, w0, inputs, outputs, rewards, alpha_scale=1.):
    jnp_hyperparams = jnp.concatenate([hyperparams[key].reshape(-1) for key in hyperparams.keys()])
    hess_fn = jax.hessian(compute_nll)
    hessian = hess_fn(jnp_hyperparams, w0, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=True)
    eigvals = jnp.linalg.eigvalsh(hessian)
    # backup_eigvals = jnp.linalg.eigvals(hessian)
    # print(eigvals)
    # print(backup_eigvals)
    return hessian, eigvals 


def optax_fit_params(init_hyperparams, w0, inputs, outputs, rewards, max_iters=2_000, alpha_scale=1., tol=1e-5):
    meta_loss_fn = partial(compute_nll, w0=w0, inputs=inputs, outputs=outputs, rewards=rewards, alpha_scale=alpha_scale, hyperparams_is_list=False)
    opt = optax.lbfgs() #(memory_size=50)
    value_and_grad_fun = optax.value_and_grad_from_state(meta_loss_fn)

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
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iters) & (err >= tol))

    init_carry = (init_hyperparams, opt.init(init_hyperparams))
    final_hyperparams, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_hyperparams, final_state


def sophisticated_fit_params(init_hyperparams, w0, inputs, outputs, rewards, max_iters=100, alpha_scale=1.):
    jnp_hyperparams = jnp.concatenate([init_hyperparams[key].reshape(-1) for key in init_hyperparams.keys()])
    meta_loss_fn = partial(compute_nll, w0=w0, inputs=inputs, outputs=outputs, rewards=rewards, alpha_scale=alpha_scale, hyperparams_is_list=True)
    x0 = jnp_hyperparams
    N = len(x0)

    solver = ScipyMinimize(
        fun=meta_loss_fn,
        method="BFGS",
        options={"disp": True},
        maxiter=max_iters
    )
    res = solver.run(init_params=x0)

    opt_hyperparams = res.params
    return opt_hyperparams


def update_hyperparams(init_hyperparams, w0, inputs, outputs, rewards, n_iters=100, lr=1e-3, alpha_scale=1.):

    @jit
    def _update_step(hyperparams):
        total_nll, meta_grads = value_and_grad(compute_nll)(hyperparams, w0, inputs, outputs, rewards, alpha_scale)
        hyperparams = jax.tree_util.tree_map(lambda p, g: jnp.maximum(p - lr * g, 0.0), hyperparams, meta_grads)
        return hyperparams, total_nll
    
    total_nlls = []
    curr_hyperparams = init_hyperparams
    for i in range(n_iters):
        curr_hyperparams, total_nll = _update_step(curr_hyperparams)

        if (i+1) % 100 == 0:
            print(f'Iter {i+1}. Total NLL: {total_nll.item()}')
        total_nlls.append(total_nll)
    # curr_hyperparams, total_nlls = lax.scan(_update_step, init_hyperparams, None, length=n_iters)

    return curr_hyperparams, total_nlls


def run_randomized_optimization_procedure(w0, inputs, outputs, rewards, key,  
                                          num_seeds=25, max_iters=500,
                                          alpha_init_scale=1e-4, beta_init_scale=1.):

    min_nll = None 
    best_hyperparams_arr = None
    for i in range(num_seeds):
        _, key = jr.split(key, 2)
        subkeys = jr.split(key, 6)

        beta0 = beta_init_scale * jnp.abs(jr.normal(subkeys[1], (1,)).astype(jnp.float32)) #jnp.array([beta_star]) + 5 * jr.normal(subkeys[1], (1,)) 
        alpha_inits = alpha_init_scale * jnp.abs(jr.normal(subkeys[2], (1,))) #jnp.array([gd_W1_alpha_star]) + 0.0029 * 2 * (jr.uniform(subkeys[2], (1,)) - 0.5)

        beta0 = jnp.log(beta0)
        alpha_inits = jnp.log(alpha_inits)

        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

        init_hyperparams = {'alpha' : alpha_inits[0], 'beta' : beta0[0]}

        # hyperparams_arr = None 
        hyperparams, _ = optax_fit_params(init_hyperparams, w0_copy, inputs, outputs, rewards, max_iters=max_iters, alpha_scale=1., tol=1e-5)
        hyperparams_arr = turn_pytree_to_arr(hyperparams)
        # hyperparams_arr = sophisticated_fit_params(init_hyperparams, w0_copy, inputs, outputs, rewards, max_iters=max_iters, alpha_scale=1.)

        # hyperparams = {'alpha' : hyperparams_arr[0], 'beta' : hyperparams_arr[1]}
        inferred_nll, _ = run_forward_weight_trajectory(hyperparams_arr, w0_copy, inputs, outputs, rewards, alpha_scale=1., hyperparams_is_list=True)

        print(f'Iter {i+1} Inferred NLL: {inferred_nll}')

        if ((min_nll is None and not jnp.isnan(inferred_nll)) or (not jnp.isnan(inferred_nll) and inferred_nll < min_nll)):
            min_nll = inferred_nll
            best_hyperparams_arr = hyperparams_arr

    return min_nll, best_hyperparams_arr


def main():
    key = jr.PRNGKey(264)    # 29 (clean), 65/68/69 (clean), 30 (clean), 31 (clean), 46, 49, 38, 45, 5783124
    _, key, skey, sskey, ssskey, sssskey = jr.split(key, 6)

    num_sessions = 40

    expert_threshold = 0.7
    model_type = 'deep'
    loss_type = 'gd_and_te'

    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session

    inputs, targets, mask = generate_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, jr.PRNGKey(11))

    init_params, (beta_star, alpha_star) = random_init_params(skey, 3, 3, 2, default_single_loss=True)
    w0 = jax.tree_util.tree_map(lambda x: jnp.array(x), init_params)
    w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

    print(f'True learning rate: {alpha_star}')
    print(f'True inverse temperature: {beta_star}')

    gd_alpha_star = 0.003
    te_alpha_star = 0.

    beta_star = 7.0

    params, losses, results_dict = fit_deep_net(init_params, num_trials, inputs, targets, None, beta_star, sskey, loss_type=loss_type,
                                                end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels,
                                                nonneg_constraint=False, gd_te_lrs=(gd_alpha_star, gd_alpha_star, te_alpha_star, te_alpha_star))

    outputs = results_dict['outputs']
    rewards = results_dict['rewards']

    eff_num_trials = results_dict['eff_num_trials']
    rightward_statistics = results_dict['rightward_statistics']
    partial_rightward_statistics = results_dict['partial_rightward_statistics']

    model_agnostic_bias = calculate_model_agnostic_bias(rightward_statistics, z)
    partial_agnostic_bias = calculate_model_agnostic_bias(partial_rightward_statistics, z)
    print(f'\nModel-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
    print(f'\nModel-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

    # create copies of the data sequence
    inputs = jnp.array(inputs)
    outputs = jnp.array(outputs)
    rewards = jnp.array(rewards)

    print(inputs.shape)
    print(rewards.shape)
    print(outputs.shape)

    accs = calculate_session_accs(rewards, num_samples_per_session, mask)
    print(f'\nMean Accuracy by Trial: \n{accs}')
    if accs[:34].max() < expert_threshold:
        print('Failed to reach expert status!')
        print(1./0)

    subkeys = jr.split(sssskey, 6)

    alpha_scale = 1. 

    min_nll, hyperparams_arr = run_randomized_optimization_procedure(w0, inputs, outputs, rewards, key,  
                                          num_seeds=3, max_iters=500,
                                          alpha_init_scale=1e-4, beta_init_scale=1.) #sophisticated_fit_params(init_hyperparams, w0, inputs, outputs, rewards)
    hyperparams = {'alpha' : hyperparams_arr[0], 'beta' : hyperparams_arr[1]}

    # w0_inferred = {'W1_diag' : hyperparams['W1_diag'], 'W2' : hyperparams['W2']}

    print(f'\nTrue GD learning rate: {gd_alpha_star}')
    print(f'True inverse temperature: {beta_star}')
    print(f'Model-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
    print(f'Model-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

    # print(f'\nInitial GD learning rate: {alpha_gd0 * alpha_scale}')
    # print(f'Initial inverse temperature: {beta0}')

    print(f'\nEstimated GD learning rate: {jnp.exp(hyperparams['alpha'])}')
    print(f'Estimated inverse temperature: {jnp.exp(hyperparams['beta'])}')

    # print(f'\nTrue init params: {init_params}')
    # print(f'Estimated init params: {w0_inferred}')
    
    # plt.figure()
    # plt.plot(np.arange(n_iters), total_nlls, lw=3, color='blue')
    # plt.xlabel('Iteration of Inference', fontsize=16)
    # plt.ylabel('NLL', fontsize=16)
    # plt.show()
    # plt.savefig('basic_alpha_beta_inference/sample_NLLs_leftbias.png')


    # Compare inferred weight trajectories with true ones
    true_hyperparams = {'alpha' : jnp.log(gd_alpha_star), 'beta' : jnp.log(beta_star)}
    inferred_hyperparams = hyperparams

    true_nll, true_ws = run_forward_weight_trajectory(true_hyperparams, w0, inputs, outputs, rewards, alpha_scale=alpha_scale)

    inferred_nll, inferred_ws = run_forward_weight_trajectory(inferred_hyperparams, w0_copy, inputs, outputs, rewards, alpha_scale=alpha_scale)

    true_ws_arr = flatten_params(true_ws)
    inferred_ws_arr = flatten_params(inferred_ws)

    print(f'True NLL: {true_nll}')
    print(f'Inferred NLL: {inferred_nll}')

    fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)

    i = 0
    for ax in axs.ravel():
        ax.plot(true_ws_arr[:, i], lw=3, color='green', label='True Weight')
        ax.plot(inferred_ws_arr[:, i], lw=1.8, linestyle='dashed', color='black', alpha=0.8, label='Inferred Weight')
        i += 1
    plt.show()
    # plt.savefig('basic_alpha_beta_inference/sample_inferred_weight_trajectories_leftbias.png')

def visualize_loss_landscape():
    alpha_star = 0.003
    beta_star = 7.0
    alpha_scale = 1.
    custom_vmax = 10_000
    num_inits = 25
    num_pts = 51

    key = jr.PRNGKey(7393)    
    _, key, skey, sskey, ssskey, sssskey = jr.split(key, 6)

    num_sessions = 40

    expert_threshold = 0.7
    model_type = 'deep'
    loss_type = 'gd_and_te'

    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session

    inputs, targets, mask = generate_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, jr.PRNGKey(11))

    init_params, (_, _) = random_init_params(skey, 3, 3, 2, default_single_loss=True)

    params, losses, results_dict = fit_deep_net(init_params, num_trials, inputs, targets, None, beta_star, sskey, loss_type=loss_type,
                                                end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels,
                                                nonneg_constraint=False, gd_te_lrs=(alpha_star, alpha_star, 0., 0.))

    outputs = results_dict['outputs']
    rewards = results_dict['rewards']

    eff_num_trials = results_dict['eff_num_trials']
    rightward_statistics = results_dict['rightward_statistics']
    partial_rightward_statistics = results_dict['partial_rightward_statistics']

    model_agnostic_bias = calculate_model_agnostic_bias(rightward_statistics, z)
    partial_agnostic_bias = calculate_model_agnostic_bias(partial_rightward_statistics, z)
    print(f'\nModel-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
    print(f'\nModel-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

    # create copies of the data sequence
    w0 = jax.tree_util.tree_map(lambda x: jnp.array(x), init_params)
    inputs = jnp.array(inputs)
    outputs = jnp.array(outputs)
    rewards = jnp.array(rewards)


    alphas = np.linspace(0., 0.025, num=num_pts, endpoint=True)
    alpha_spacing = alphas[1] - alphas[0]
    alpha_star_idx = alpha_star / alpha_spacing
    

    betas = np.linspace(0., 10., num=num_pts, endpoint=True)
    beta_spacing = betas[1] - betas[0]
    beta_star_idx = beta_star / beta_spacing 

    nll_matrix = np.zeros((num_pts, num_pts))
    for i in range(len(alphas)):
        alpha = alphas[i]
        for j in range(len(betas)):
            beta = betas[j]

            w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

            curr_hyperparams = {'alpha' : jnp.log(alpha), 'beta' : jnp.log(beta)}
            nll, _ = run_forward_weight_trajectory(curr_hyperparams, w0_copy, inputs, outputs, rewards, alpha_scale=alpha_scale)
            nll_matrix[i, j] = nll
    
    (true_min_x, true_min_y) = np.unravel_index(np.argmin(nll_matrix, axis=None), nll_matrix.shape)
    print(true_min_x, true_min_y)

    min_alpha, min_beta = true_min_x * alpha_spacing, true_min_y * beta_spacing

    print(nll_matrix)

    estimated_hyperparams = np.zeros((2, num_inits))
    key = jr.PRNGKey(1984)
    for k in range(num_inits):
        _, key = jr.split(key, 2)
        subkeys = jr.split(key, 4)

        beta0 = 1. * jnp.abs(jr.normal(subkeys[1], (1,)))
        alpha_gd0 = 1e-4 * jnp.abs(jr.normal(subkeys[2], (1,)))
        alpha_scale = 1.
        init_hyperparams = {'alpha' : jnp.log(alpha_gd0[0]), 'beta' : jnp.log(beta0[0])}

        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

        # hyperparams_arr = sophisticated_fit_params(init_hyperparams, w0_copy, inputs, outputs, rewards, max_iters=500, alpha_scale=1.)
        # hyperparams = {'alpha' : hyperparams_arr[0], 'beta' : hyperparams_arr[1]}
        hyperparams, _ = optax_fit_params(init_hyperparams, w0_copy, inputs, outputs, rewards, max_iters=500, alpha_scale=1., tol=1e-5)
        # hyperparams_arr = turn_pytree_to_arr(hyperparams)

        inferred_nll, _ = run_forward_weight_trajectory(hyperparams, w0_copy, inputs, outputs, rewards, alpha_scale=alpha_scale)
        eigvals = None 
        if not jnp.isnan(inferred_nll):
            _, eigvals = compute_hessian(hyperparams, w0_copy, inputs, outputs, rewards, alpha_scale=alpha_scale)
        print(f'Iter {k+1} done. Hessian eigvals: {eigvals}')

        estimated_hyperparams[0, k] = jnp.exp(hyperparams['alpha'])
        estimated_hyperparams[1, k] = jnp.exp(hyperparams['beta'])

    print(estimated_hyperparams)



    # Display the matrix as an image
    plt.figure(figsize=(15, 15))
    plt.imshow(nll_matrix, cmap='viridis', vmin=np.min(nll_matrix), vmax=np.max(nll_matrix))
    plt.colorbar()
    plt.scatter([beta_star_idx], [alpha_star_idx], color='red', s=200)
    plt.scatter([true_min_y], [true_min_x], color='black', s=200)
    plt.scatter(estimated_hyperparams[1]/beta_spacing, estimated_hyperparams[0]/alpha_spacing, color='blue', s=100)
    plt.xticks(np.arange(num_pts)[::5], betas[::5])
    plt.yticks(np.arange(num_pts)[::5], alphas[::5])
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.imshow(nll_matrix, cmap='viridis', vmin=np.min(nll_matrix), vmax=5900)
    plt.colorbar()
    plt.scatter([beta_star_idx], [alpha_star_idx], color='red', s=200)
    plt.scatter([true_min_y], [true_min_x], color='black', s=200)
    plt.scatter(estimated_hyperparams[1]/beta_spacing, estimated_hyperparams[0]/alpha_spacing, color='blue', s=100)
    plt.xticks(np.arange(num_pts)[::5], betas[::5])
    plt.yticks(np.arange(num_pts)[::5], alphas[::5])
    plt.show()


    print(f'Min NLL found by grid search: {np.min(nll_matrix)}') 

    true_hyperparams = {'alpha' : jnp.log(alpha_star), 'beta' : jnp.log(beta_star)}
    true_nll, true_ws = run_forward_weight_trajectory(true_hyperparams, w0, inputs, outputs, rewards, alpha_scale=alpha_scale)
    print(f'True data NLL: {true_nll}')

    candidate_alphas = [0.0025, 0.003, 0.0035, 0.004]
    candidate_betas = [9., 7., 5., 4.]

    for m in range(len(candidate_alphas)):
        # test_alphas = np.linspace(alpha_star, min_alpha, num=50)
        curr_alpha = candidate_alphas[m]
        curr_beta = candidate_betas[m]
        test_betas = np.linspace(curr_beta - 1, curr_beta + 1, num=50)
        test_nlls = np.zeros_like(test_betas)
        for i in range(len(test_betas)):
            beta = test_betas[i]

            w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

            hparams = {'alpha' : jnp.log(curr_alpha), 'beta' : jnp.log(beta)}
            nll, _ = run_forward_weight_trajectory(hparams, w0_copy, inputs, outputs, rewards, alpha_scale=alpha_scale)
            test_nlls[i] = nll 
        
        print(f'NLLs along a line segment (in beta direction) centered at (alpha = {curr_alpha}, beta = {curr_beta}): \n{test_nlls}')


def visualize_log_loss_landscape():
    alpha_star = 0.003
    beta_star = 7.0
    alpha_scale = 1.
    custom_vmax = 10_000
    num_inits = 50

    key = jr.PRNGKey(7393)    
    _, key, skey, sskey, ssskey, sssskey = jr.split(key, 6)

    num_sessions = 40

    expert_threshold = 0.7
    model_type = 'deep'
    loss_type = 'gd_and_te'

    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session

    inputs, targets, mask = generate_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, jr.PRNGKey(11))

    init_params, (_, _) = random_init_params(skey, 3, 3, 2, default_single_loss=True)

    params, losses, results_dict = fit_deep_net(init_params, num_trials, inputs, targets, None, beta_star, sskey, loss_type=loss_type,
                                                end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels,
                                                nonneg_constraint=False, gd_te_lrs=(alpha_star, alpha_star, 0., 0.))

    outputs = results_dict['outputs']
    rewards = results_dict['rewards']

    eff_num_trials = results_dict['eff_num_trials']
    rightward_statistics = results_dict['rightward_statistics']
    partial_rightward_statistics = results_dict['partial_rightward_statistics']

    model_agnostic_bias = calculate_model_agnostic_bias(rightward_statistics, z)
    partial_agnostic_bias = calculate_model_agnostic_bias(partial_rightward_statistics, z)
    print(f'\nModel-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
    print(f'\nModel-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

    # create copies of the data sequence
    w0 = jax.tree_util.tree_map(lambda x: jnp.array(x), init_params)
    inputs = jnp.array(inputs)
    outputs = jnp.array(outputs)
    rewards = jnp.array(rewards)

    num_pts = 26

    # alphas = np.linspace(0., 0.025, num=num_pts, endpoint=True)
    # alpha_spacing = alphas[1] - alphas[0]
    # alpha_star_idx = alpha_star / alpha_spacing
    log_alpha_start = np.log(1e-7)
    log_alphas = np.linspace(log_alpha_start, np.log(0.025), num=num_pts, endpoint=True)
    log_alpha_spacing = log_alphas[1] - log_alphas[0]
    alpha_star_idx = (np.log(alpha_star) - log_alpha_start) / log_alpha_spacing
    

    # betas = np.linspace(0., 10., num=num_pts, endpoint=True)
    # beta_spacing = betas[1] - betas[0]
    # beta_star_idx = beta_star / beta_spacing 
    log_beta_start = np.log(1e-7)
    log_betas = np.linspace(log_beta_start, np.log(10.), num=num_pts, endpoint=True)
    log_beta_spacing = log_betas[1] - log_betas[0]
    beta_star_idx = (np.log(beta_star) - log_beta_start) / log_beta_spacing

    nll_matrix = np.zeros((num_pts, num_pts))
    for i in range(len(log_alphas)):
        log_alpha = log_alphas[i]
        for j in range(len(log_betas)):
            log_beta = log_betas[j]

            w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

            curr_hyperparams = {'alpha' : log_alpha, 'beta' : log_beta}
            nll, _ = run_forward_weight_trajectory(curr_hyperparams, w0_copy, inputs, outputs, rewards, alpha_scale=alpha_scale)
            nll_matrix[i, j] = nll
    
    (true_min_x, true_min_y) = np.unravel_index(np.argmin(nll_matrix, axis=None), nll_matrix.shape)
    print(true_min_x, true_min_y)

    min_log_alpha, min_log_beta = true_min_x * log_alpha_spacing + log_alpha_start, true_min_y * log_beta_spacing + log_beta_start

    print(nll_matrix)


    estimated_hyperparams = np.zeros((2, num_inits))
    key = jr.PRNGKey(1984)
    for k in range(num_inits):
        _, key = jr.split(key, 2)
        subkeys = jr.split(key, 4)

        beta0 = 1. * jnp.abs(jr.normal(subkeys[1], (1,)))
        alpha_gd0 = 1e-4 * jnp.abs(jr.normal(subkeys[2], (1,)))
        alpha_scale = 1.
        init_hyperparams = {'alpha' : jnp.log(alpha_gd0), 'beta' : jnp.log(beta0)}

        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

        hyperparams_arr = sophisticated_fit_params(init_hyperparams, w0_copy, inputs, outputs, rewards, max_iters=500, alpha_scale=1.)
        hyperparams = {'alpha' : hyperparams_arr[0], 'beta' : hyperparams_arr[1]}

        inferred_nll, _ = run_forward_weight_trajectory(hyperparams, w0_copy, inputs, outputs, rewards, alpha_scale=alpha_scale)
        eigvals = None 
        if not jnp.isnan(inferred_nll):
            _, eigvals = compute_hessian(hyperparams, w0_copy, inputs, outputs, rewards, alpha_scale=alpha_scale)
        print(f'Iter {k+1} done. Hessian eigvals: {eigvals}')

        estimated_hyperparams[0, k] = hyperparams['alpha']
        estimated_hyperparams[1, k] = hyperparams['beta']

    print(estimated_hyperparams)

    # Display the matrix as an image
    plt.figure(figsize=(15, 15))
    plt.imshow(nll_matrix, cmap='viridis', vmin=np.min(nll_matrix), vmax=np.max(nll_matrix))
    plt.colorbar()
    plt.scatter([beta_star_idx], [alpha_star_idx], color='red', s=200)
    plt.scatter([true_min_y], [true_min_x], color='black', s=200)
    plt.scatter((estimated_hyperparams[1] - log_beta_start)/log_beta_spacing, (estimated_hyperparams[0]-log_alpha_start)/log_alpha_spacing, color='blue', s=100)
    # plt.xticks(np.arange(num_pts)[::5], log_betas[::5])
    # plt.yticks(np.arange(num_pts)[::5], log_alphas[::5])
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.imshow(nll_matrix, cmap='viridis', vmin=np.min(nll_matrix), vmax=10_000)
    plt.colorbar()
    plt.scatter([beta_star_idx], [alpha_star_idx], color='red', s=200)
    plt.scatter([true_min_y], [true_min_x], color='black', s=200)
    plt.scatter((estimated_hyperparams[1] - log_beta_start)/log_beta_spacing, (estimated_hyperparams[0]-log_alpha_start)/log_alpha_spacing, color='blue', s=100)
    # plt.xticks(np.arange(num_pts)[::5], log_betas[::5])
    # plt.yticks(np.arange(num_pts)[::5], log_alphas[::5])
    plt.show()



if __name__=="__main__":
    # main()
    visualize_loss_landscape()
    # visualize_log_loss_landscape()








