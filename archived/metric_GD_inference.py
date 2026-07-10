# --- path shim: allow running this legacy script standalone after the move to legacy/ ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path shim ---
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax.tree_util import tree_map
import optax
import optax.tree_utils as otu
from functools import partial
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import simple_rates_and_weights_inference
from simple_rates_and_weights_inference import generate_train_test_split, run_randomized_optimization_procedure
from shallow_linear_net import shallow_net, shallow_net_params, fit_shallow_model
from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from utils import calculate_model_agnostic_bias, calculate_session_accs, flatten_params, turn_pytree_to_arr, realign_weight_trajectories, combine_batch_weights, stratified_quantile_split_indices, compute_two_layer_flattened_net_params, generate_randomized_multisession_graded_data
from metric_GD_network import *


def run_metric_GD_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list):
    """
    Arguments:
    - hyperparams (alphas, beta, w0)
    - inputs
    - outputs
    - rewards
    - learning rule
    - hyperparams_is_list (True or False)

    Returns:
    - total negative log-likelihood of the data given the hyperparams (alphas, beta)
    """
    assert hyperparams_is_list is True or hyperparams_is_list is False

    if hyperparams_is_list:
        alpha_W_const = hyperparams[0]
        alpha_W_stim = hyperparams[1]
        beta = hyperparams[2]
        W = hyperparams[3:9].reshape(2, 3)
    else:
        alpha_W_const = hyperparams['alpha_const']  
        alpha_W_stim = hyperparams['alpha_stim']
        beta = hyperparams['beta']
        W = hyperparams['W']

    alphas = jnp.array([alpha_W_const, alpha_W_stim, alpha_W_stim]).reshape(1, -1)
    lrs_W = {'W' : alphas}
    
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
    def _apply_updates(w, grads, log_lrs):
        return jax.tree_util.tree_map(lambda p, g, log_lr: p - jnp.exp(log_lr) * g, w, grads, log_lrs)

    def _step(w, data_point):
        x, y, rew = data_point

        _, gd_grads = value_and_grad(_squared_error_loss_fn)(w, x, y, rew)
        w_new = _apply_updates(w, gd_grads, lrs_W)

        out = shallow_net(w, x)
        ll = _log_likelihood(out, y)

        return w_new, (ll, w_new)

    # inputs_and_outputs_and_rewards = jnp.concatenate((inputs, outputs.reshape(-1, 1), rewards.reshape(-1, 1)), axis=-1)    # num_trials x (input_dim (3) + output_dim (1))
    data_seq = (inputs, outputs, rewards)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    total_nll = -jnp.sum(lls)

    return total_nll, ws 


def run_metric_GD_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list):
    """
    Arguments:
    - hyperparams (alphas, beta, w0)
    - inputs
    - outputs
    - rewards

    Returns:
    - total negative log-likelihood of the data given the hyperparams (alphas, beta)
    """
    assert hyperparams_is_list is True or hyperparams_is_list is False


    if hyperparams_is_list:
        alpha_W1 = hyperparams[0]
        alpha_W2_const = hyperparams[1]
        alpha_W2_stim = hyperparams[2]
        beta = hyperparams[3]
        W1_diag_0 = hyperparams[4:7]
        W2_0 = hyperparams[7:13].reshape(2, 3)
    else:
        alpha_W1 = hyperparams['alpha_W1']  
        alpha_W2_const = hyperparams['alpha_W2_const']
        alpha_W2_stim = hyperparams['alpha_W2_stim']
        beta = hyperparams['beta']
        W1_diag_0 = hyperparams['W1_diag_0']
        W2_0 = hyperparams['W2_0']

    # Use same learning rate (alpha_W2_stim) for both stimulus-related weights
    alphas_W2 = jnp.array([alpha_W2_const, alpha_W2_stim, alpha_W2_stim]).reshape(1, -1)

    lrs_gd = {
        "W1_diag": alpha_W1,
        "W2": alphas_W2
    }
    
    w0 = {'W1_diag' : W1_diag_0, 'W2' : W2_0}

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
    def _apply_updates(w, grads, log_lrs):
        return jax.tree_util.tree_map(lambda p, g, log_lr: p - jnp.exp(log_lr) * g, w, grads, log_lrs)

    def _step(w, data_point):
        x, y, rew = data_point

        # compute GD gradients 
        _, gd_grads = value_and_grad(_squared_error_loss_fn)(w, x, y, rew)

        # now, apply diagonal/metric GD updates
        w_new = _apply_updates(w, gd_grads, lrs_gd)

        # for the current weights compute the log-likelihood of the data given the hyperparams (alpha, beta)
        out, _ = cortico_striatal_net(w, x)
        ll = _log_likelihood(out, y)

        return w_new, (ll, w_new)

    # inputs_and_outputs_and_rewards = jnp.concatenate((inputs, outputs.reshape(-1, 1), rewards.reshape(-1, 1)), axis=-1)    # num_trials x (input_dim (3) + output_dim (1))
    data_seq = (inputs, outputs, rewards)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    total_nll = -jnp.sum(lls)

    return total_nll, ws 

# following functions operate on batches of (inputs, outputs, rewards), and return a batched array of total nlls and batched pytree of weight trajectories
batched_run_metric_GD_deep_forward_weight_trajectory = vmap(run_metric_GD_deep_forward_weight_trajectory, in_axes=(None, 0, 0, 0, None))
batched_run_metric_GD_onelayer_forward_weight_trajectory = vmap(run_metric_GD_onelayer_forward_weight_trajectory, in_axes=(None, 0, 0, 0, None))


def run_metric_GD_forward_trajectory(hyperparams, inputs, outputs, rewards, model_type, batched, hyperparams_is_list=True):
    ws = None 
    nll = None 
    if batched:
        if model_type == 'two-layer':
            nll, ws = batched_run_metric_GD_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list=hyperparams_is_list)
        else:
            nll, ws = batched_run_metric_GD_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list=hyperparams_is_list)
    else:
        if model_type == 'two-layer':
            nll, ws = run_metric_GD_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list=hyperparams_is_list)
        else:
            nll, ws = run_metric_GD_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list=hyperparams_is_list)

    nll = jnp.mean(nll)

    return nll, ws


def compute_nll(hyperparams, inputs, outputs, rewards, hyperparams_is_list=True, model_type='two-layer'):
    assert model_type in ['two-layer', 'one-layer']

    total_nll = None
    if model_type == 'two-layer':
        total_nll, _ = run_metric_GD_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list)
        # w0 = hyperparams[2:]
        # total_nll + 1e-4 * jnp.sum(w0**2)
    else:
        total_nll, _ = run_metric_GD_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list)

    return total_nll


def compute_batched_nll(hyperparams, inputs, outputs, rewards, hyperparams_is_list=True, model_type='two-layer'):
    assert model_type in ['two-layer', 'one-layer']

    all_nlls = None
    if model_type == 'two-layer':
        all_nlls, _ = batched_run_metric_GD_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list)
    else:
        all_nlls, _ = batched_run_metric_GD_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list)

    batch_nll = jnp.mean(all_nlls)

    return batch_nll


def compute_hessian(hyperparams_arr, inputs, outputs, rewards, model_type='two-layer', batched=False):
    assert model_type in ['two-layer', 'one-layer']
    hess_fn = None 
    hessian = None 
    if batched:
        hess_fn = jax.hessian(compute_batched_nll)
        hessian = hess_fn(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, model_type=model_type)
    else:
        hess_fn = jax.hessian(compute_nll)
        hessian = hess_fn(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, model_type=model_type)
    
    # hessian = jacfwd(jacrev(compute_nll))(jnp_hyperparams, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=True)
    eigvals = jnp.linalg.eigvalsh(hessian)
    return hessian, eigvals 


def optax_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=500, 
                     tol=1e-4, model_type='two-layer', batched=False):
    assert model_type in ['two-layer', 'one-layer']

    jnp_hyperparams = jnp.concatenate([init_hyperparams[key].reshape(-1) for key in init_hyperparams.keys()])
    meta_loss_fn = None 
    if batched:
        meta_loss_fn = partial(compute_batched_nll, inputs=inputs, outputs=outputs, rewards=rewards, hyperparams_is_list=True, model_type=model_type)
    else:
        meta_loss_fn = partial(compute_nll, inputs=inputs, outputs=outputs, rewards=rewards, hyperparams_is_list=True, model_type=model_type)
    opt = optax.lbfgs(memory_size=50) 
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


def run_randomized_metric_GD_inference_procedure(inputs, outputs, rewards, key, num_seeds=25, max_iters=500, tol=1e-4,
                                          alpha_init_scale=1e-1, beta_init_scale=1e0, w0_init_scale=1e-1,
                                          model_type='two-layer', batched=False,
                                          alpha_inits_guess=None, beta_init_guess=None, w0_init_guess=None, noise_level=0.1):
    assert model_type in ['two-layer', 'one-layer']

    all_hyperparams = []
    all_nlls = []

    min_nll = None 
    best_w0 = None 
    best_hyperparams_arr = None
    for i in range(num_seeds):
        _, key = jr.split(key, 2)
        subkeys = jr.split(key, 10)

        beta0 = beta_init_scale * jnp.abs(jr.normal(subkeys[1], (1,)).astype(jnp.float32))
        if beta_init_guess is not None:
            beta0 = beta_init_guess * (1 + noise_level * jr.normal(subkeys[5], (1,)))

        alpha_inits = alpha_init_scale * jnp.abs(jr.normal(subkeys[2], (3,))) 
        if alpha_inits_guess is not None:
            alpha_inits = alpha_inits_guess * (1 + noise_level * jr.normal(subkeys[6], (3,)))

        beta0 = jnp.log(beta0)
        alpha_inits = jnp.log(alpha_inits)

        init_hyperparams = None 
        if model_type == 'two-layer':
            W1_diag_0 = w0_init_scale * jr.normal(subkeys[3], (3,)) 
            W2_0 = w0_init_scale * jr.normal(subkeys[4], (2, 3)) 
            if w0_init_guess is not None:
                W1_diag_0 = w0_init_guess['W1_diag'] * (1 + noise_level * jr.normal(subkeys[7], (3,)))
                W2_0 = w0_init_guess['W2'] * (1 + noise_level * jr.normal(subkeys[8], (2, 3)))

            init_hyperparams = {'alpha_W1' : alpha_inits[0], 'alpha_W2_const' : alpha_inits[1], 'alpha_W2_stim' : alpha_inits[2], 
                                'beta' : beta0[0], 'W1_diag_0' : W1_diag_0, 'W2_0' : W2_0}
        else:
            W_0 = w0_init_scale * jr.normal(subkeys[4], (2, 3)) 
            if w0_init_guess is not None:
                W_0 = w0_init_guess['W'] * (1 + noise_level * jr.normal(subkeys[8], (2, 3)))
            init_hyperparams = {'alpha_const' : alpha_inits[1], 'alpha_stim' : alpha_inits[2], 'beta' : beta0[0], 'W' : W_0}

        hyperparams_arr, _ = optax_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=max_iters, 
                                              tol=tol, model_type=model_type, batched=batched)

        hyperparams = None
        if model_type == 'two-layer':
            hyperparams = {'alpha_W1' : hyperparams_arr[0], 'alpha_W2_const' : hyperparams_arr[1], 'alpha_W2_stim' : hyperparams_arr[2], 
                           'beta' : hyperparams_arr[3], 'W1_diag_0' : hyperparams_arr[4:7], 'W2_0' : hyperparams_arr[7:13].reshape(2, 3)}
        else:
            hyperparams = {'alpha_const' : hyperparams_arr[0], 'alpha_stim' : hyperparams_arr[1], 'beta' : hyperparams_arr[2], 'W' : hyperparams_arr[3:9].reshape(2, 3)}

        inferred_nll = None 
        if batched:
            inferred_nll = compute_batched_nll(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, model_type=model_type)
        else:
            inferred_nll = compute_nll(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, model_type=model_type)

        eigvals = None 
        grad = None
        if not jnp.isnan(inferred_nll):
            _, eigvals = compute_hessian(hyperparams_arr, inputs, outputs, rewards, model_type=model_type, batched=batched)

            if batched:
                grad = jax.grad(compute_batched_nll)(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, model_type=model_type)
            else:
                grad = jax.grad(compute_nll)(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, model_type=model_type)

            all_hyperparams.append(hyperparams_arr)
            all_nlls.append(inferred_nll)

        # \nNearby NLLs: {nearby_nlls}
        print(f'Iter {i+1} Inferred NLL: {inferred_nll} \nGradient: {grad} \nHessian eigvals: {eigvals}')
        if model_type == 'two-layer':
            print(f'Inferred alphas: {jnp.exp(hyperparams_arr[0:3])}, Inferred beta: {jnp.exp(hyperparams_arr[3])}')
            print(f'Inferred W1_diag_0: {hyperparams["W1_diag_0"]}, Inferred W2_0: {hyperparams["W2_0"]}')
        else:
            print(f'Inferred alphas: {jnp.exp(hyperparams_arr[0:2])}, Inferred beta: {jnp.exp(hyperparams_arr[2])}')
            print(f'Inferred W_0: {hyperparams["W"]}')
        print('\n\n')

        if ((min_nll is None and not jnp.isnan(inferred_nll)) or (not jnp.isnan(inferred_nll) and inferred_nll < min_nll)):
            min_nll = inferred_nll
            if model_type == 'two-layer':
                best_w0 = {'W1_diag' : hyperparams['W1_diag_0'], 'W2' : hyperparams['W2_0']}
            else:
                best_w0 = {'W' : hyperparams['W']}
            best_hyperparams_arr = hyperparams_arr

    return best_w0, min_nll, best_hyperparams_arr, (jnp.array(all_hyperparams), jnp.array(all_nlls))


def visualize_inferred_vs_true_effective_weights(inferred_hyperparams_arr, true_hyperparams_arr, inputs, outputs, rewards, 
                                                 true_model_type, test_model_type, plot_both_left_right=True, is_tutor_executor=False):
    if is_tutor_executor:
        true_nll, true_ws = simple_rates_and_weights_inference.run_deep_forward_weight_trajectory(true_hyperparams_arr, inputs, outputs, rewards, 'te', hyperparams_is_list=True)
    else:
        true_nll, true_ws = run_metric_GD_forward_trajectory(true_hyperparams_arr, inputs, outputs, rewards, true_model_type, batched=False, hyperparams_is_list=True)

    inferred_nll, inferred_ws = run_metric_GD_forward_trajectory(inferred_hyperparams_arr, inputs, outputs, rewards, test_model_type, batched=False, hyperparams_is_list=True)

    fig, axs = plt.subplots(1 + plot_both_left_right, 3, figsize=(18, 5*(1+plot_both_left_right)), sharex=True, sharey=True)

    true_trajectories = flatten_params(true_ws)
    inferred_trajectories = flatten_params(inferred_ws)
    if true_model_type == 'two-layer':
        true_trajectories = compute_two_layer_flattened_net_params(true_trajectories)
    
    if test_model_type == 'two-layer':
        inferred_trajectories = compute_two_layer_flattened_net_params(inferred_trajectories)

    j = 0
    for ax in axs.ravel():
        if plot_both_left_right:
            ax.plot(inferred_trajectories[:, j], lw=3., color='black', linestyle='dashed')
            ax.plot(true_trajectories[:, j], lw=3., color='green')
        else:
            ax.plot(inferred_trajectories[:, j] - inferred_trajectories[:, j+3], lw=3., color='black', linestyle='dashed')
            ax.plot(true_trajectories[:, j] - true_trajectories[:, j+3], lw=3., color='green')
        ax.axhline(0, lw=2., color='blue', linestyle=':')
        j += 1

    plt.show()

    print(f'True Data NLL: {true_nll}')
    print(f'Inferred NLL: {inferred_nll}')

    return


def PCA_visualization(inferred_hyperparams_collection, nlls, NLL_threshold):
    '''
    Arguments: 
    - inferred_hyperparams_collection: list of inferred hyperparams arrays from different random seeds (num_seeds x num_hyperparams)
    - nlls: list of NLL values corresponding to each inferred hyperparams array
    - NLL_threshold: only include inferred hyperparams arrays with NLL below this threshold
    '''

    filtered_hyperparams = []
    for i in range(len(nlls)):
        if nlls[i] <= NLL_threshold:
            filtered_hyperparams.append(inferred_hyperparams_collection[i])
    
    filtered_hyperparams = jnp.array(filtered_hyperparams)
    print(len(filtered_hyperparams))
    # print(filtered_hyperparams)

    # first, compute proportion of variance explained by each PC without any standardization
    pca = PCA()
    pca.fit(filtered_hyperparams)
    var_explained = pca.explained_variance_ratio_
    print('Proportion of variance explained by each PC (no standardization):')
    print(var_explained)

    print(f'Cumulative variance explained:')
    print(jnp.cumsum(var_explained))

    print('Eigenvalues: ')
    eigvals = pca.explained_variance_

    plt.figure()
    plt.scatter(np.arange(1, len(eigvals)+1), eigvals, s=100)
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.show()

    print('Eigenvectors: ')
    print(pca.components_)

    top_eigvecs = pca.components_[0:2, :]
    plt.figure()
    plt.stem(np.arange(1, len(eigvals)+1), top_eigvecs[0], linefmt='r-', markerfmt='ro', basefmt='r-', label='1st Eigvec')
    plt.stem(np.arange(1, len(eigvals)+1), top_eigvecs[1], linefmt='g-', markerfmt='go', basefmt='g-', label='2nd Eigvec')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvector Coefficient')
    plt.legend()
    plt.show()

    print('\n')

    # second, standardize the hyperparams and repeat
    # standardized_hyperparams = (filtered_hyperparams - jnp.mean(filtered_hyperparams, axis=0)) / jnp.std(filtered_hyperparams, axis=0)
    # pca_std = PCA()
    # pca_std.fit(standardized_hyperparams)
    # var_explained_std = pca_std.explained_variance_ratio_
    # print('Proportion of variance explained by each PC (with standardization):')
    # print(var_explained_std)

    # print('Eigenvalues: ')
    # print(pca_std.explained_variance_)

    # print('Eigenvectors: ')
    # print(pca_std.components_)

    return filtered_hyperparams

def PCA_exploration():
    all_inferred_hyperparams = np.load('metric_GD_PCA_exploration/metric_GD_inferred_hyperparams_nsamples100.npy')
    all_nlls = np.load('metric_GD_PCA_exploration/metric_GD_inferred_nlls_nsamples100.npy')
    true_nll = 5645.

    filtered_hyperparams = PCA_visualization(all_inferred_hyperparams, all_nlls, NLL_threshold=true_nll)

def main():
    true_model_type = 'two-layer'
    test_model_type = 'two-layer'
    key = jr.PRNGKey(34)
    _, key = jr.split(key, 2)
    skeys = jr.split(key, 6)

    num_sessions = 40

    expert_threshold = 0.7

    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session
    session_info = (num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast)

    # inputs, targets, zero_contrast_mask = generate_randomized_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, 
    #                                                                                    num_zero_contrast, skeys[1])

    # init_params, _, _ = random_init_params(skeys[2], lr_scale=0.001, lr_mean=0.003, beta_scale=1.0, beta_mean=10.0)

    beta = 10.
    W1_lr = 0.003
    W2_const_lr = 0.002
    W2_stim_lr = 0.006

    lrs = jnp.array([W1_lr, W2_const_lr, W2_stim_lr])


    # noisy_lrs = lrs + 0.001 * jr.normal(skeys[1], lrs.shape)
    # noisy_beta = beta + 0.2 * jr.normal(skeys[2], (1,))

    # w0 = jax.tree_util.tree_map(lambda x: jnp.array(x), init_params)
    # w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

    # Train for a modest number of iterations (this calls the training loop inside fit_GD_deep_net)

    # trained_params, losses, results = fit_GD_deep_net(init_params, num_trials=num_trials, num_sessions=num_sessions,
    #                                                   inputs=inputs, targets=targets, lrs=lrs, beta=beta,
    #                                                   zero_contrast_mask=zero_contrast_mask, key=skeys[3],
    #                                                   nonneg_constraint=False, end_condition=False, num_grading_levels=num_grading_levels)

    # one_layer_lrs = jnp.array([W2_const_lr, W2_stim_lr])

    num_train_sets = 3
    num_test_sets = 3
    datasets, train_test_dict = generate_metric_GD_train_test_split(lrs, beta, true_model_type, session_info, skeys[3], num_train_sets=num_train_sets, num_test_sets=num_test_sets,
                                                                    vary_inputs=False, true_w0=None, do_bias_stratified_split=False, nonneg_init_weights=True)
    
    # te_alpha_star = 0.005
    # te_beta_star = 7.

    # datasets, train_test_dict = generate_train_test_split(te_alpha_star, te_beta_star, 'te', 'two-layer', session_info, skeys[3], 
    #                                                       num_train_sets=num_train_sets, num_test_sets=num_test_sets, vary_inputs=False)
    

    (train_inputs, train_outputs, train_rewards, train_w0) = datasets[0]
    batched_train_inputs = train_test_dict['train_inputs']
    batched_train_outputs = train_test_dict['train_outputs']
    batched_train_rewards = train_test_dict['train_rewards']

    ground_truth_test_biases = train_test_dict['test_ma_biases']
    
    print(train_test_dict['train_pma_biases'])
    print(train_test_dict['train_ma_biases'])
    print('\n')

    # noisy_train_w0 = {
    #     'W1_diag' : train_w0['W1_diag'] * (1. + 0.05 * jr.normal(skeys[4], train_w0['W1_diag'].shape)),
    #     'W2' : train_w0['W2'] * (1. + 0.05 * jr.normal(skeys[4], train_w0['W2'].shape))
    # }

    # inputs = jnp.array(inputs, dtype=jnp.float32)
    # outputs = jnp.asarray(results['outputs'])
    # rewards = jnp.asarray(results['rewards'], dtype=jnp.float32)

    # Summaries to plot: session accuracies, zero-contrast bias, slope differences, stim DA biases
    # Accuracy per session (uses utility function)
    # num_samples_per_session = num_trials // num_sessions
    # accs = calculate_session_accs(np.array(results['rewards']), num_samples_per_session, zero_contrast_mask)
    # print(accs)

    # biases = np.array(results['biases'])
    # slope_diffs = np.array(results['slope_diffs'])
    # stim_DA_biases = np.array(results['stim_DA_biases'])

    # rightward_stats = results['rightward_statistics']
    # partial_rightward_stats = results['partial_rightward_statistics']
    # ma = calculate_model_agnostic_bias(rightward_stats, z)
    # pma = calculate_model_agnostic_bias(partial_rightward_stats, z)
    # print(pma)
    # print(ma)



    # inference pipeline begins here

    true_hyperparams_arr = jnp.concatenate([jnp.log(jnp.concatenate([lrs, jnp.array([beta])])), train_w0['W1_diag'], train_w0['W2'].reshape(-1)])
    true_nll = compute_batched_nll(true_hyperparams_arr, batched_train_inputs, batched_train_outputs, batched_train_rewards, hyperparams_is_list=True, model_type=true_model_type)

    # true_hyperparams_arr = jnp.concatenate([jnp.log(jnp.concatenate([one_layer_lrs, jnp.array([beta])])), train_w0['W'].reshape(-1)])
    # true_nll = compute_batched_nll(true_hyperparams_arr, batched_train_inputs, batched_train_outputs, batched_train_rewards, hyperparams_is_list=True, model_type=true_model_type)

    # true_hyperparams_arr = jnp.concatenate([jnp.log(jnp.array([te_alpha_star, te_beta_star])), train_w0['W1_diag'], train_w0['W2'].reshape(-1)])
    # true_nll = simple_rates_and_weights_inference.compute_batched_nll(true_hyperparams_arr, batched_train_inputs, batched_train_outputs, batched_train_rewards, hyperparams_is_list=True, model_type='two-layer', learning_rule='te')


    # true_nll = compute_nll(true_hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, model_type=true_model_type)
    print(f'True Batch-averaged Data NLL: {true_nll}')

    # num_alpha_blocks = 1
    # l2_lambda = 0.  # 1.
    # hyperparams_is_list = True 
    # w0_inferred, min_nll, hyperparams_arr = run_randomized_optimization_procedure(batched_train_inputs, batched_train_outputs, batched_train_rewards, skeys[5], learning_rule='gd',
    #                                                                               num_seeds=15, max_iters=500, 
    #                                                                               tol=1e-4, method='optax',
    #                                                                               alpha_init_scale=0.01, beta_init_scale=1., w0_init_scale=0.1, 
    #                                                                               num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda, batched=True, model_type=test_model_type)
    
    # alpha_inits_guess = jnp.repeat(jnp.exp(hyperparams_arr[0]), 3)
    # beta_init_guess = jnp.exp(hyperparams_arr[1])
    # w0_init_guess = w0_inferred

    w0_inferred, min_nll, inferred_hyperparams_arr, (all_hyperparams, all_nlls) = run_randomized_metric_GD_inference_procedure(batched_train_inputs, batched_train_outputs, batched_train_rewards, skeys[4], num_seeds=15,
                                                                                           max_iters=500, tol=1e-4, alpha_init_scale=0.01, beta_init_scale=10., w0_init_scale=0.1,
                                                                                           model_type=test_model_type, batched=True)
                                                                                           # alpha_inits_guess=alpha_inits_guess, beta_init_guess=beta_init_guess, w0_init_guess=w0_init_guess, noise_level=0.05)
    
    # print(len(all_nlls))
    # n_samples = all_nlls.shape[0]
    # print(all_hyperparams.shape)
    # print(all_nlls.shape)
    # np.save(f'metric_GD_PCA_exploration/metric_GD_inferred_hyperparams_nsamples{n_samples}_groundtruthlrs_{W1_lr}_{W2_const_lr}_{W2_stim_lr}_beta{beta}.npy', all_hyperparams)
    # np.save(f'metric_GD_PCA_exploration/metric_GD_inferred_nlls_nsamples{n_samples}_groundtruthlrs_{W1_lr}_{W2_const_lr}_{W2_stim_lr}_beta{beta}.npy', all_nlls)

    # epsilon = 2.0
    # filtered_hyperparams = PCA_visualization(all_hyperparams, all_nlls, NLL_threshold=min_nll+epsilon)
    
    print(f'Minimum NLL obtained on training data: {min_nll}')
    
    print(f'True learning rates: {lrs}')

    # print(f'Model-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
    # print(f'Model-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

    # print(f'\nInitial learning rates: {alpha_inits * alpha_scale}')
    # print(f'Initial inverse temperature: {beta0}')
    # print(f'Initial starting weights: {W1_diag_0, W2_0}')

    print(f'\nEstimated learning rate(s): {jnp.exp(inferred_hyperparams_arr[:3])}')
    print(f'Estimated inverse temperature: {jnp.exp(inferred_hyperparams_arr[3])}')

    print(f'\nTrue init params: {train_w0}')
    print(f'Estimated init params: {w0_inferred}')


    for i in range(num_train_sets + num_test_sets):
        (test_inputs, test_outputs, test_rewards, _) = datasets[i]  # num_train_sets+i
        visualize_inferred_vs_true_effective_weights(inferred_hyperparams_arr, true_hyperparams_arr, test_inputs, test_outputs, test_rewards, 
                                                 true_model_type, test_model_type, plot_both_left_right=True, is_tutor_executor=False)


        visualize_inferred_vs_true_effective_weights(inferred_hyperparams_arr, true_hyperparams_arr, test_inputs, test_outputs, test_rewards, 
                                                 true_model_type, test_model_type, plot_both_left_right=False, is_tutor_executor=False)







if __name__=="__main__":
    main()
    # PCA_exploration()
