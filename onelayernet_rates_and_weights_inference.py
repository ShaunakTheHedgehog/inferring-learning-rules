import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax import debug
from jax.tree_util import tree_map
import optax
import optax.tree_utils as otu
import jaxopt
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
# from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from simple_rates_and_weights_inference import run_deep_forward_weight_trajectory
from shallow_linear_net import shallow_net_params, fit_shallow_model, shallow_net
from deep_linear_net import random_init_params, fit_deep_net, cortico_striatal_net
from utils import generate_multisession_graded_data, calculate_model_agnostic_bias, calculate_session_accs, flatten_params, realign_weight_trajectories

def turn_pytree_to_arr(tree):
    arr = jnp.concatenate([tree[key].reshape(-1) for key in tree.keys()])
    return arr

def run_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, learning_rule, hyperparams_is_list):
    """
    Arguments:
    - hyperparams (alpha, beta, w0)
    - inputs
    - outputs
    - rewards

    Returns:
    - total negative log-likelihood of the data given the hyperparams (alphas, beta)
    """
    assert learning_rule in ['gd', 'te']
    assert hyperparams_is_list is True or hyperparams_is_list is False

    if hyperparams_is_list:
        alpha = hyperparams[0]
        beta = hyperparams[1]
        W = hyperparams[2:8].reshape(2, 3)
    else:
        alpha = hyperparams['alpha']  
        beta = hyperparams['beta']
        W = hyperparams['W']
    
    w0 = {'W' : W}

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
        out = shallow_net(w, x)
        Q_y = out[y_choice]
        return 0.5 * (Q_y - rew)**2
    
    @jit
    def _stim_loss(w, x, y_choice, rew):
        W = w["W"]               # (2, 3)

        Q_stim = jnp.dot(W[y_choice, 1:], x[1:])
        loss = 0.5 * (rew - Q_stim)**2
        return loss

    @jit
    def _const_loss(w, x, y_choice, rew):
        W = w["W"]               # (2, 3)

        Q_const = W[y_choice, 0] * x[0]
        loss = 0.5 * (rew - Q_const)**2
        return loss
    
    @jit 
    def _tutor_executor_loss_fn(w, x, y_choice, rew):
        return _stim_loss(w, x, y_choice, rew) + _const_loss(w, x, y_choice, rew)
    
    @jit 
    def _apply_updates(w, grads, lr):
        return jax.tree_util.tree_map(lambda p, g: p - jnp.exp(lr) * g, w, grads)

    def _step(w, data_point):
        x, y, rew = data_point

        if learning_rule == 'gd':
            _, gd_grads = value_and_grad(_squared_error_loss_fn)(w, x, y, rew)
            w_new = _apply_updates(w, gd_grads, alpha)
        else:
            _, te_grads = jax.value_and_grad(_tutor_executor_loss_fn)(w, x, y, rew)
            w_new = _apply_updates(w, te_grads, alpha)

        out = shallow_net(w, x)
        ll = _log_likelihood(out, y)

        return w_new, (ll, w_new)

    # inputs_and_outputs_and_rewards = jnp.concatenate((inputs, outputs.reshape(-1, 1), rewards.reshape(-1, 1)), axis=-1)    # num_trials x (input_dim (3) + output_dim (1))
    data_seq = (inputs, outputs, rewards)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    total_nll = -jnp.sum(lls)

    return total_nll, ws 


def compute_nll(hyperparams, inputs, outputs, rewards, hyperparams_is_list=False, learning_rule='gd'):
    total_nll, _ = run_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list=hyperparams_is_list, learning_rule=learning_rule)
    return total_nll

def compute_hessian(hyperparams, inputs, outputs, rewards, learning_rule='gd'):
    assert learning_rule in ['gd', 'te']
    jnp_hyperparams = jnp.concatenate([hyperparams[key].reshape(-1) for key in hyperparams.keys()])
    hess_fn = jax.hessian(compute_nll)
    hessian = hess_fn(jnp_hyperparams, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=learning_rule)
    eigvals = jnp.linalg.eigvalsh(hessian)
    return hessian, eigvals 

def epsilon_ball_test(hyperparams_arr, inputs, outputs, rewards, key, learning_rule='gd', num_samples=25, epsilon=0.01):
    assert learning_rule in ['gd', 'te']
    N = len(hyperparams_arr)
    curr_nll, _ = run_onelayer_forward_weight_trajectory(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=learning_rule)
    nearby_nlls = []
    for _ in range(num_samples):
        _, key = jr.split(key, 2)
        perturbed_hyperparams = hyperparams_arr + jr.normal(key, (N,)) * epsilon 
        inferred_nll, _ = run_onelayer_forward_weight_trajectory(perturbed_hyperparams, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=learning_rule)
        nearby_nlls.append(inferred_nll.item())
    return curr_nll, nearby_nlls


def optax_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=500, tol=1e-6, learning_rule='gd'):
    assert learning_rule in ['gd', 'te']

    jnp_hyperparams = jnp.concatenate([init_hyperparams[key].reshape(-1) for key in init_hyperparams.keys()])
    meta_loss_fn = partial(compute_nll, inputs=inputs, outputs=outputs, rewards=rewards, hyperparams_is_list=True, learning_rule=learning_rule)
    opt = optax.lbfgs(memory_size=100) #(memory_size=50)
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

    init_carry = (jnp_hyperparams, opt.init(jnp_hyperparams))
    final_hyperparams, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_hyperparams, final_state


def sophisticated_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=2_000, learning_rule='gd'):
    assert learning_rule in ['gd', 'te']
    jnp_hyperparams = jnp.concatenate([init_hyperparams[key].reshape(-1) for key in init_hyperparams.keys()])
    meta_loss_fn = partial(compute_nll, inputs=inputs, outputs=outputs, rewards=rewards, hyperparams_is_list=True, learning_rule=learning_rule)
    x0 = jnp_hyperparams
    N = len(x0)

    # solver = ScipyMinimize(
    #     fun=meta_loss_fn,
    #     method="BFGS",
    #     options={"disp": True},
    #     maxiter=max_iters
    # )
    # res = solver.run(init_params=x0)

    solver = BFGS(
        fun=meta_loss_fn,
        tol=1e-6,           # Tighter function tolerance
        maxiter=max_iters
    )
    
    res = solver.run(init_params=x0)
    opt_hyperparams = res.params

    return opt_hyperparams


def run_randomized_optimization_procedure(inputs, outputs, rewards, key, learning_rule='gd', 
                                          num_seeds=25, max_iters=500, method='optax', tol=1e-6,
                                          alpha_init_scale=1e-1, beta_init_scale=1e0, w0_init_scale=1e-1,
                                          epsilon=0.001):
    assert learning_rule in ['gd', 'te']
    assert method in ['jaxopt', 'optax']

    min_nll = None 
    best_w0 = None 
    best_hyperparams_arr = None
    for i in range(num_seeds):
        _, key = jr.split(key, 2)
        subkeys = jr.split(key, 6)

        beta0 = beta_init_scale * jnp.abs(jr.normal(subkeys[1], (1,)).astype(jnp.float32)) #jnp.array([beta_star]) + 5 * jr.normal(subkeys[1], (1,)) 
        alpha_inits = alpha_init_scale * jnp.abs(jr.normal(subkeys[2], (1,))) #jnp.array([gd_W1_alpha_star]) + 0.0029 * 2 * (jr.uniform(subkeys[2], (1,)) - 0.5)

        beta0 = jnp.log(beta0)
        alpha_inits = jnp.log(alpha_inits)

        W_0 = w0_init_scale * jr.normal(subkeys[4], (2, 3)) # w0['W2'] + 0.1 * jr.normal(subkeys[4], (2, 3))   # 

        init_hyperparams = {'alpha' : alpha_inits[0], 'beta' : beta0[0], 'W' : W_0}

        hyperparams_arr = None 
        if method == 'jaxopt':
            hyperparams_arr = sophisticated_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=max_iters, learning_rule=learning_rule)
        else:
            hyperparams_arr, _ = optax_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=max_iters, tol=tol, learning_rule=learning_rule)

        hyperparams = {'alpha' : hyperparams_arr[0], 'beta' : hyperparams_arr[1], 'W' : hyperparams_arr[2:8].reshape(2, 3)}
        inferred_nll, _ = run_onelayer_forward_weight_trajectory(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=learning_rule)

        _, nearby_nlls = epsilon_ball_test(hyperparams_arr, inputs, outputs, rewards, subkeys[5], learning_rule=learning_rule, num_samples=25, epsilon=epsilon)

        eigvals = None 
        grad = None
        if not jnp.isnan(inferred_nll):
            _, eigvals = compute_hessian(hyperparams, inputs, outputs, rewards, learning_rule=learning_rule)
            grad = jax.grad(compute_nll)(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=learning_rule)

        print(f'Iter {i+1} Inferred NLL: {inferred_nll} \nNearby NLLs: {nearby_nlls} \nGradient: {grad} \nHessian eigvals: {eigvals}')
        print('\n\n')

        if ((min_nll is None and not jnp.isnan(inferred_nll)) or (not jnp.isnan(inferred_nll) and inferred_nll < min_nll)):
            min_nll = inferred_nll
            best_w0 = {'W' : hyperparams['W']}
            best_hyperparams_arr = hyperparams_arr

    return best_w0, min_nll, best_hyperparams_arr

def main():
    key = jr.PRNGKey(905)   
    _, key, skey, sskey, ssskey, sssskey = jr.split(key, 6)

    num_sessions = 40

    expert_threshold = 0.7
    # model_type = 'deep'
    loss_type = 'gd_and_te'

    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session

    inputs, targets, mask = generate_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, jr.PRNGKey(11))

    init_params, (beta_star, _) = random_init_params(skey, 3, 3, 2, default_single_loss=True)
    w0 = jax.tree_util.tree_map(lambda x: jnp.array(x), init_params)
    w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
    # print(f'True learning rate: {alpha_star}')
    # print(f'True inverse temperature: {beta_star}')

    gd_rate = 0.
    te_rate = 0.003

    gd_W1_alpha_star = gd_rate + te_rate
    gd_W2_alpha_star = gd_rate
    te_W2const_alpha_star = te_rate
    te_W2stim_alpha_star = te_rate

    true_learning_rule = 'te'
    test_learning_rule = 'gd'

    alpha_star = None 
    if true_learning_rule == 'gd':
        alpha_star = gd_rate
        assert te_rate == 0.
    else:
        alpha_star = te_rate
        assert gd_rate == 0.

    beta_star = 10.00

    params, losses, results_dict = fit_deep_net(init_params, num_trials, inputs, targets, None, beta_star, sskey, loss_type=loss_type,
                                                end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels,
                                                nonneg_constraint=False, gd_te_lrs=(gd_W1_alpha_star, gd_W2_alpha_star, te_W2const_alpha_star, te_W2stim_alpha_star))

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
    inputs = jnp.array(inputs, dtype=jnp.float32)
    outputs = jnp.array(outputs)
    rewards = jnp.array(rewards, dtype=jnp.float32)

    print(inputs.shape)
    print(rewards.shape)
    print(outputs.shape)

    accs = calculate_session_accs(rewards, num_samples_per_session, mask)
    print(f'\nMean Accuracy by Trial: \n{accs}')
    if accs[:34].max() < expert_threshold:
        print('Failed to reach expert status!')
        # print(1./0)

    subkeys = jr.split(sssskey, 6)

    print(f'Testing the {test_learning_rule} learning rule within a ONE LAYER NET...')

    true_hyperparams = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), w0['W1_diag'], w0['W2'].reshape(-1)])
    true_nll, true_ws = run_deep_forward_weight_trajectory(true_hyperparams, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=true_learning_rule)
    print(f'\nTrue Data NLL with {true_learning_rule} (in a DEEP net) as the true learning rule: {true_nll}')

    hyperparams_is_list = True 
    w0_inferred, min_nll, hyperparams_arr = run_randomized_optimization_procedure(inputs, outputs, rewards, subkeys[5], learning_rule=test_learning_rule,
                                                                                  num_seeds=30, max_iters=500, 
                                                                                  tol=1e-6, method='optax',
                                                                                  alpha_init_scale=0.1, beta_init_scale=1., w0_init_scale=0.1, epsilon=0.001)
    
    hyperparams = {'alpha' : hyperparams_arr[0], 'beta' : hyperparams_arr[1], 'W' : hyperparams_arr[2:8].reshape(2, 3)}

    #hyperparams_arr = jnp.array([hyperparams[key].item() for key in hyperparams.keys()])
    # w0_inferred = {'W1_diag' : hyperparams['W1_diag_0'], 'W2' : hyperparams['W2_0']}

    print(f'\nTrue GD W learning rate in deep net: {gd_rate}')
    print(f'True TE W learning rate in deep net: {te_rate}')
    print(f'True inverse temperature in deep net: {beta_star}')

    print(f'\nModel-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
    print(f'Model-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

    print(f'\nEstimated {test_learning_rule} learning rate for one layer net: {jnp.exp(hyperparams['alpha'])}')
    print(f'Estimated inverse temperature for one layer net: {jnp.exp(hyperparams['beta'])}')

    print(f'\nTrue init params: {w0_copy}')
    print(f'Estimated init params: {w0_inferred}')


    # Compare inferred weight trajectories with true ones
    hyperparams_is_list=True 
    true_hyperparams = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), w0['W1_diag'], w0['W2'].reshape(-1)])  
    inferred_hyperparams = turn_pytree_to_arr(hyperparams) #hyperparams_arr #hyperparams

    true_nll, true_ws = run_deep_forward_weight_trajectory(true_hyperparams, inputs, outputs, rewards, hyperparams_is_list=hyperparams_is_list, learning_rule=true_learning_rule)

    inferred_nll, inferred_ws = run_onelayer_forward_weight_trajectory(inferred_hyperparams, inputs, outputs, rewards, hyperparams_is_list=hyperparams_is_list, learning_rule=test_learning_rule)

    true_ws_arr = flatten_params(true_ws)
    inferred_ws_arr = flatten_params(inferred_ws)

    print(true_ws_arr.shape)
    print(inferred_ws_arr.shape)

    print(f'True NLL: {true_nll}')
    print(f'Inferred NLL: {inferred_nll}')

    # fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)

    # i = 0
    # for ax in axs.ravel():
    #     ax.plot(true_ws_arr[:, i], lw=3, color='green', label='True Weight')
    #     ax.plot(inferred_ws_arr[:, i], lw=1.8, linestyle='dashed', color='black', alpha=0.8, label='Inferred Weight')
    #     i += 1
    # # plt.show()
    # plt.savefig(f'testing_baseline_onelayer_nets/og_sample_true_{true_learning_rule}_inferring_{test_learning_rule}.png')
    

    
    # fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)

    # realigned_inferred_ws = realign_weight_trajectories(true_ws_arr, inferred_ws_arr)

    # i = 0
    # for ax in axs.ravel():
    #     ax.plot(true_ws_arr[:, i], lw=3, color='green', label='True Weight')
    #     ax.plot(realigned_inferred_ws[:, i], lw=1.8, linestyle='dashed', color='black', alpha=0.8, label='Inferred Weight')
    #     i += 1
    # # plt.show()

    # plt.savefig(f'testing_baseline_onelayer_nets/sample_true_{true_learning_rule}_inferring_{test_learning_rule}.png')



def visualize_loss_landscape():
    alpha_star = 0.01
    beta_star = 12.0
    alpha_scale = 0.001
    custom_vmax = 10_000
    num_inits = 100

    key = jr.PRNGKey(264)    
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

    num_pts = 51

    alphas = np.linspace(0., 0.5, num=num_pts, endpoint=True)
    alpha_spacing = alphas[1] - alphas[0]
    alpha_star_idx = alpha_star / alpha_spacing

    betas = np.linspace(0., 25., num=num_pts, endpoint=True)
    beta_spacing = betas[1] - betas[0]
    beta_star_idx = beta_star / beta_spacing 

    nll_matrix = np.zeros((num_pts, num_pts))
    for i in range(len(alphas)):
        alpha = alphas[i]
        for j in range(len(betas)):
            beta = betas[j]

            w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

            curr_hyperparams = {'alpha' : alpha / alpha_scale, 'beta' : beta}
            nll, _ = run_onelayer_forward_weight_trajectory(curr_hyperparams, w0_copy, inputs, outputs, rewards, alpha_scale=alpha_scale)
            nll_matrix[i, j] = nll
    
    (true_min_x, true_min_y) = np.unravel_index(np.argmin(nll_matrix, axis=None), nll_matrix.shape)
    print(true_min_x, true_min_y)

    print(nll_matrix)

    estimated_hyperparams = np.zeros((2, num_inits))
    key = jr.PRNGKey(947)
    for k in range(num_inits):
        _, key = jr.split(key, 2)
        subkeys = jr.split(key, 4)

        beta0 = 0.01 * jnp.abs(jr.normal(subkeys[1], (1,)).astype(jnp.float32))
        alpha0 = 0.01 * jnp.abs(jr.normal(subkeys[2], (1,))).astype(jnp.float32)
        alpha_scale = 0.001
        init_hyperparams = {'alpha' : alpha0, 'beta' : beta0}

        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

        hyperparams_arr = sophisticated_fit_params(init_hyperparams, w0_copy, inputs, outputs, rewards)
        hyperparams = {'alpha' : hyperparams_arr[0], 'beta' : hyperparams_arr[1]}

        estimated_hyperparams[0, k] = hyperparams['alpha'] * alpha_scale
        estimated_hyperparams[1, k] = hyperparams['beta']

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
    plt.imshow(nll_matrix, cmap='viridis', vmin=np.min(nll_matrix), vmax=custom_vmax)
    plt.colorbar()
    plt.scatter([beta_star_idx], [alpha_star_idx], color='red', s=200)
    plt.scatter([true_min_y], [true_min_x], color='black', s=200)
    plt.scatter(estimated_hyperparams[1]/beta_spacing, estimated_hyperparams[0]/alpha_spacing, color='blue', s=100)
    plt.xticks(np.arange(num_pts)[::5], betas[::5])
    plt.yticks(np.arange(num_pts)[::5], alphas[::5])
    plt.show()



if __name__=="__main__":
    main()
    # visualize_loss_landscape()






