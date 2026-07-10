# --- path shim: allow running this legacy script standalone after the move to legacy/ ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path shim ---
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
from shallow_linear_net import shallow_net
from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from utils import generate_multisession_graded_data, calculate_model_agnostic_bias, calculate_session_accs, flatten_params, turn_pytree_to_arr, realign_weight_trajectories
from simple_rates_and_weights_inference import * 


if __name__=="__main__":
    key = jr.PRNGKey(405)   
    _, key, skey, sskey, ssskey = jr.split(key, 5)

    te_rate = jnp.arange(0.01, 0.0009, -0.001)
    gd_rate = jnp.zeros(10)
    beta_star = jnp.array([10., 9., 11., 8., 12., 7., 13., 6., 14., 10.]) 

    true_learning_rule = 'te'
    num_sessions = 40
    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session
    loss_type = 'gd_and_te'

    gd_W1_alpha_star = gd_rate + te_rate
    gd_W2_alpha_star = gd_rate
    te_W2const_alpha_star = te_rate
    te_W2stim_alpha_star = te_rate

    print(gd_W2_alpha_star.shape)
    print(te_W2const_alpha_star.shape)
    print(te_W2stim_alpha_star.shape)
    # session_info = (num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast)

    # [train_data, test_data_with_w0] = generate_train_test_split(alpha_star, beta_star, true_learning_rule, session_info, key)

    # (train_inputs, train_outputs, train_rewards, train_w0) = train_data 
    # (test_inputs, test_outputs, test_rewards, _) = test_data_with_w0
    # test_data = (test_inputs, test_outputs, test_rewards)

    inputs, targets, mask = generate_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, jr.PRNGKey(11))

    init_params, (beta_star, alpha_star) = random_init_params(skey, 3, 3, 2, default_single_loss=True)
    w0 = jax.tree_util.tree_map(lambda x: jnp.array(x), init_params)
    w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)


    params, losses, results_dict = fit_deep_net(init_params, num_trials, inputs, targets, None, beta_star, sskey, loss_type=loss_type,
                                                end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels,
                                                nonneg_constraint=False, gd_te_lrs=(gd_W1_alpha_star, gd_W2_alpha_star, te_W2const_alpha_star, te_W2stim_alpha_star), num_lr_blocks=10)
    outputs = results_dict['outputs']
    rewards = results_dict['rewards']

    inputs = jnp.array(inputs)
    outputs = jnp.array(outputs)
    rewards = jnp.array(rewards)

    num_alpha_blocks = 10
    l2_lambda = 1.  # 1.
    hyperparams_is_list = True 
    test_learning_rule = 'te'
    w0_inferred, min_nll, hyperparams_arr = run_randomized_optimization_procedure(inputs, outputs, rewards, ssskey, learning_rule=test_learning_rule,
                                                                                  num_seeds=15, max_iters=1_000, 
                                                                                  tol=1e-6, method='optax',
                                                                                  alpha_init_scale=0.1, beta_init_scale=1., w0_init_scale=0.1, 
                                                                                  num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda)
    hyperparams = {'alpha' : hyperparams_arr[:num_alpha_blocks], 'beta' : hyperparams_arr[num_alpha_blocks], 'W1_diag_0' : hyperparams_arr[num_alpha_blocks+1:num_alpha_blocks+4], 'W2_0' : hyperparams_arr[num_alpha_blocks+4:num_alpha_blocks+10].reshape(2, 3)}


    hyperparams_is_list=True 
    true_hyperparams = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), w0_copy['W1_diag'], w0_copy['W2'].reshape(-1)])  # {'alpha' : gd_alpha_star / alpha_scale, 'alpha2' : te_alpha_star / alpha_scale, 'beta' : beta_star, 'W1_diag' : init_hyperparams['W1_diag'], 'W2' : init_hyperparams['W2']}
    inferred_hyperparams = turn_pytree_to_arr(hyperparams) #hyperparams_arr #hyperparams

    true_nll, true_ws = run_deep_forward_weight_trajectory(true_hyperparams, inputs, outputs, rewards, hyperparams_is_list=hyperparams_is_list, learning_rule=true_learning_rule)

    inferred_nll, inferred_ws = run_blocked_deep_forward_weight_trajectory(inferred_hyperparams, inputs, outputs, rewards, hyperparams_is_list=hyperparams_is_list, learning_rule=test_learning_rule, num_blocks=num_alpha_blocks)

    true_ws_arr = flatten_params(true_ws)
    inferred_ws_arr = flatten_params(inferred_ws)

    print(f'True NLL on train set: {true_nll}')
    print(f'Inferred NLL on train set: {inferred_nll}')

    fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)

    i = 0
    for ax in axs.ravel():
        ax.plot(true_ws_arr[:, i], lw=3, color='green', label='True Weight')
        ax.plot(inferred_ws_arr[:, i], lw=1.8, linestyle='dashed', color='black', alpha=0.8, label='Inferred Weight')
        i += 1
    # plt.show()
