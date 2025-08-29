import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax import debug
from jax.tree_util import tree_map
import optax
import jaxopt 
from jaxopt import ScipyMinimize, ScipyBoundedMinimize
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
from utils import generate_multisession_graded_data, calculate_model_agnostic_bias, calculate_session_accs, flatten_params
# from simple_multi_learning_rule_inference import * 
from simple_rates_and_weights_inference import run_randomized_optimization_procedure, run_deep_forward_weight_trajectory, turn_pytree_to_arr

def inference_sweep_by_trials(num_sessions_arr, num_repeats, w0, gd_W1_alpha_star, gd_W2_alpha_star, 
                              te_W2const_alpha_star, te_W2stim_alpha_star, beta_star, key, num_grading_levels = 4,
                              num_samples_per_grading = 30, num_zero_contrast = 10, alpha_scale=1.):
    
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    z = 1.*num_zero_contrast / num_samples_per_session

    max_sessions = jnp.max(num_sessions_arr).item()

    inputs, targets, mask = generate_multisession_graded_data(max_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, jr.PRNGKey(11))

    hyperparams_matrix = np.zeros((len(num_sessions_arr), num_repeats, 11))
    inferred_nlls = np.zeros((len(num_sessions_arr), num_repeats))
    true_nlls = np.zeros_like(inferred_nlls)

    for i in range(len(num_sessions_arr)):
        num_sessions = num_sessions_arr[i]
        num_trials = num_sessions * num_samples_per_session
        for j in range(num_repeats):
            _, key = jr.split(key, 2)
            skeys = jr.split(key, 6)

            curr_inputs = inputs[:num_trials]
            curr_targets = targets[:num_trials]
            curr_mask = mask[:num_trials]

            params = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

            _, _, results_dict = fit_deep_net(params, num_trials, curr_inputs, curr_targets, None, beta_star, skeys[1], loss_type='gd_and_te',
                                                end_condition=True, zero_contrast_mask=curr_mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels,
                                                nonneg_constraint=False, gd_te_lrs=(gd_W1_alpha_star, gd_W2_alpha_star, te_W2const_alpha_star, te_W2stim_alpha_star))

            curr_outputs = results_dict['outputs']
            curr_rewards = results_dict['rewards']

            eff_num_trials = results_dict['eff_num_trials']
            rightward_statistics = results_dict['rightward_statistics']
            partial_rightward_statistics = results_dict['partial_rightward_statistics']

            model_agnostic_bias = calculate_model_agnostic_bias(rightward_statistics, z)
            partial_agnostic_bias = calculate_model_agnostic_bias(partial_rightward_statistics, z)
            print(f'\nModel-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
            print(f'Model-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

            # create copies of the data sequence
            # w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
            # w0_copy2 = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)

            curr_inputs = jnp.array(curr_inputs, dtype=jnp.float32)
            curr_outputs = jnp.array(curr_outputs)
            curr_rewards = jnp.array(curr_rewards, dtype=jnp.float32)

            accs = calculate_session_accs(curr_rewards, num_samples_per_session, curr_mask)
            print(f'\nMean Accuracy by Trial: \n{accs}')
            if accs[:34].max() < 0.7:
                print('Failed to reach expert status!')
                # print(1./0)

            # beta0 = 1. * jnp.abs(jr.normal(skeys[2], (1,)).astype(jnp.float32))

            # alpha_inits = 1e-4 * jnp.abs(jr.normal(skeys[3], (1,))).astype(jnp.float32)

            # # init_hyperparams = {'alpha_gd_W1' : alpha_inits[0], 'alpha_gd_W2' : alpha_inits[1], 'alpha_te_W2const' : alpha_inits[2], 'alpha_te_W2stim' : alpha_inits[3], 'beta' : beta0} 
            # init_hyperparams = {'alpha_gd' : alpha_inits[0], 'beta' : beta0[0]}
            print('\nReady for inference!\n')    

            hyperparams_is_list = True
            w0_inferred, min_nll, hyperparams_arr = run_randomized_optimization_procedure(curr_inputs, curr_outputs, curr_rewards, key, l2_lambda=0., l1_lambda=0., 
                                                                                          num_seeds=100, max_iters=500, alpha_scale=1., method='jaxopt',
                                                                                          alpha_init_scale=1e-4, beta_init_scale=1., w0_init_scale=0.1)
            
            print(f'MIN NLL: {min_nll}')
            print(f'alpha = {jnp.exp(hyperparams_arr[0])}, beta = {jnp.exp(hyperparams_arr[1])}')
            # hyperparams_arr = sophisticated_fit_params(init_hyperparams, w0_copy, curr_inputs, curr_outputs, curr_rewards, n_iters=2_000, alpha_scale=alpha_scale)
            # hyperparams = {'alpha_gd_W1' : hyperparams_arr[0], 'alpha_gd_W2' : hyperparams_arr[1], 'alpha_te_W2const' : hyperparams_arr[2], 'alpha_te_W2stim' : hyperparams_arr[3], 'beta' : hyperparams_arr[4]} 

            hyperparams_matrix[i, j] = hyperparams_arr

            # Compare inferred weight trajectories with true ones
            # true_hyperparams = jnp.concatenate([jnp.array([gd_W1_alpha_star / alpha_scale, gd_W2_alpha_star / alpha_scale, te_W2const_alpha_star / alpha_scale, te_W2stim_alpha_star / alpha_scale, beta_star])]) 
            true_hyperparams = jnp.concatenate([jnp.log(jnp.array([gd_W1_alpha_star / alpha_scale, beta_star])), w0['W1_diag'], w0['W2'].reshape(-1)])
            inferred_hyperparams = hyperparams_arr #hyperparams

            true_nll, _ = run_deep_forward_weight_trajectory(true_hyperparams, curr_inputs, curr_outputs, curr_rewards, alpha_scale=alpha_scale, hyperparams_is_list=hyperparams_is_list)

            inferred_nll, _ = run_deep_forward_weight_trajectory(inferred_hyperparams, curr_inputs, curr_outputs, curr_rewards, alpha_scale=alpha_scale, hyperparams_is_list=hyperparams_is_list)

            inferred_nlls[i, j] = inferred_nll
            true_nlls[i, j] = true_nll


            # true_ws_arr = flatten_params(true_ws)
            # inferred_ws_arr = flatten_params(inferred_ws)

            # fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)

            # i = 0
            # for ax in axs.ravel():
            #     ax.plot(true_ws_arr[:, i], lw=3, color='green', label='True Weight')
            #     ax.plot(inferred_ws_arr[:, i], lw=1.8, linestyle='dashed', color='black', alpha=0.8, label='Inferred Weight')
            #     i += 1
            # plt.show()


    return hyperparams_matrix, inferred_nlls, true_nlls

if __name__=="__main__":
    gd_W1_alpha_star = 0.002
    gd_W2_alpha_star = 0.002
    te_W2const_alpha_star = 0.00
    te_W2stim_alpha_star = 0.00
    beta_star = 7.0

    alpha_scale = 1.

    key = jr.PRNGKey(29459843)
    skeys = jr.split(key, 10)

    num_sessions_arr = jnp.arange(20, 81, 20)
    num_repeats = 1
    w0, _ = random_init_params(skeys[1], 3, 3, 2, default_single_loss=True)

    # true_hyperparams = np.array([gd_W1_alpha_star / alpha_scale, gd_W2_alpha_star / alpha_scale, te_W2const_alpha_star / alpha_scale, te_W2stim_alpha_star / alpha_scale, beta_star])
    true_hyperparams = jnp.concatenate([jnp.array([gd_W1_alpha_star, beta_star]), w0['W1_diag'], w0['W2'].reshape(-1)])

    hyperparams_matrix, inferred_nlls, true_nlls = inference_sweep_by_trials(num_sessions_arr, num_repeats, w0, gd_W1_alpha_star, 
                                                                             gd_W2_alpha_star, te_W2const_alpha_star, te_W2stim_alpha_star,
                                                                             beta_star, skeys[2], alpha_scale=alpha_scale)
    
    print(hyperparams_matrix.mean(-2))
    hyperparams_matrix[:, :, :2] = jnp.exp(hyperparams_matrix[:, :, :2])

    mses = np.mean(np.sum((hyperparams_matrix[:, :, :2] - true_hyperparams[:2].reshape(1, 1, -1))**2, axis=-1), axis=-1)
    avg_inferred_nlls = np.mean(inferred_nlls, axis=-1)
    avg_true_nlls = np.mean(true_nlls, axis=-1)

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(num_sessions_arr, mses, lw=3, color='blue')
    axs[0].set_ylabel('Mean SSE', fontsize=14)

    axs[1].plot(num_sessions_arr, avg_true_nlls, lw=3, color='green', label='True')
    axs[1].plot(num_sessions_arr, avg_inferred_nlls, lw=2., linestyle='dashed', color='black', alpha=0.8, label='Inferred')
    axs[1].set_ylabel('Mean NLL', fontsize=14)
    axs[1].set_xlabel('Number of Sessions', fontsize=14)
    axs[1].legend()

    plt.savefig('alpha_beta_convergence_over_trials_1repeat_no_log.png')
    #plt.show()