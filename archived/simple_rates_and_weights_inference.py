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
import seaborn as sns
import pandas as pd

import functools
import random
import math
from shallow_linear_net import shallow_net, shallow_net_params, fit_shallow_model
from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from utils import generate_multisession_graded_data, calculate_model_agnostic_bias, calculate_session_accs, flatten_params, turn_pytree_to_arr, realign_weight_trajectories, combine_batch_weights, stratified_quantile_split_indices, compute_two_layer_flattened_net_params, generate_randomized_multisession_graded_data


def run_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, learning_rule, hyperparams_is_list):
    """
    Arguments:
    - hyperparams (alpha, beta, w0)
    - inputs
    - outputs
    - rewards
    - learning rule
    - hyperparams_is_list (True or False)

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



def run_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, learning_rule, hyperparams_is_list, nln):
    """
    Arguments:
    - hyperparams (alphas, beta, w0)
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
        W1_diag_0 = hyperparams[2:5]
        W2_0 = hyperparams[5:11].reshape(2, 3)
    else:
        alpha = hyperparams['alpha']  
        beta = hyperparams['beta']
        W1_diag_0 = hyperparams['W1_diag_0']
        W2_0 = hyperparams['W2_0']

    lr_gd = {
        "W1_diag": alpha,
        "W2": alpha
    }

    lr_te_W2 = {
        "W1_diag": -jnp.inf, 
        "W2": alpha
    }

    lr_te_W1 = {
        "W1_diag": alpha, 
        "W2": -jnp.inf
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
        out, _ = cortico_striatal_net(w, x, nln)
        Q_y = out[y_choice]
        return 0.5 * (Q_y - rew)**2
    
    @jit
    def _stim_loss(w, x, y_choice, rew):
        W1_diag = w["W1_diag"]     # (3,)
        W2 = w["W2"]               # (2, 3)

        Q_stim = jnp.dot(W2[y_choice, 1:], nln(W1_diag[1:] * x[1:]))
        loss = 0.5 * (rew - Q_stim)**2
        return loss

    @jit
    def _const_loss(w, x, y_choice, rew):
        W1_diag = w["W1_diag"]     # (3,)
        W2 = w["W2"]               # (2, 3)

        Q_const = W2[y_choice, 0] * nln(W1_diag[0] * x[0])
        loss = 0.5 * (rew - Q_const)**2
        return loss
    
    @jit 
    def _tutor_executor_loss_fn(w, x, y_choice, rew):
        return _stim_loss(w, x, y_choice, rew) + _const_loss(w, x, y_choice, rew)
    
    @jit 
    def _apply_updates(w, grads, lrs):
        return jax.tree_util.tree_map(lambda p, g, lr: p - jnp.exp(lr) * g, w, grads, lrs)

    def _step(w, data_point):
        x, y, rew = data_point

        # compute GD gradients -- this will be used in both GD and TE rules
        _, gd_grads = value_and_grad(_squared_error_loss_fn)(w, x, y, rew)
        w_new = None

        # now, do weight updates based on either GD or TE
        if learning_rule == 'gd':
            w_new = _apply_updates(w, gd_grads, lr_gd)
        else:
            _, te_grads_W2 = jax.value_and_grad(_tutor_executor_loss_fn)(w, x, y, rew)
            w_new = _apply_updates(w, te_grads_W2, lr_te_W2)
            w_new = _apply_updates(w_new, gd_grads, lr_te_W1)

        # for the current weights compute the log-likelihood of the data given the hyperparams (alpha, beta)
        out, _ = cortico_striatal_net(w, x, nln)
        ll = _log_likelihood(out, y)

        return w_new, (ll, w_new)

    # inputs_and_outputs_and_rewards = jnp.concatenate((inputs, outputs.reshape(-1, 1), rewards.reshape(-1, 1)), axis=-1)    # num_trials x (input_dim (3) + output_dim (1))
    data_seq = (inputs, outputs, rewards)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    total_nll = -jnp.sum(lls)

    return total_nll, ws 

# following functions operate on batches of (inputs, outputs, rewards), and return a batched array of total nlls and batched pytree of weight trajectories
batched_run_deep_forward_weight_trajectory = vmap(run_deep_forward_weight_trajectory, in_axes=(None, 0, 0, 0, None, None, None))
batched_run_onelayer_forward_weight_trajectory = vmap(run_onelayer_forward_weight_trajectory, in_axes=(None, 0, 0, 0, None, None))


def run_blocked_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, learning_rule, hyperparams_is_list, num_blocks):
    """
    Arguments:
    - hyperparams (alphas, beta, w0)
    - inputs
    - outputs
    - rewards

    Returns:
    - total negative log-likelihood of the data given the hyperparams (alphas, beta)
    """
    assert learning_rule in ['gd', 'te']
    assert hyperparams_is_list is True or hyperparams_is_list is False
    assert num_blocks >= 1
    assert inputs.shape[0] % num_blocks == 0

    block_length = inputs.shape[0] // num_blocks

    if hyperparams_is_list:
        alpha = hyperparams[:num_blocks]
        beta = hyperparams[num_blocks]
        W1_diag_0 = hyperparams[num_blocks+1:num_blocks+4]
        W2_0 = hyperparams[num_blocks+4:num_blocks+10].reshape(2, 3)
    else:
        alpha = hyperparams['alpha']  
        beta = hyperparams['beta']
        W1_diag_0 = hyperparams['W1_diag_0']
        W2_0 = hyperparams['W2_0']

    alpha_arr = jnp.repeat(alpha, block_length)
    
    w0 = {'W1_diag' : W1_diag_0, 'W2' : W2_0}

    @jit
    def _get_lr_pytree(curr_alpha):
        lr_gd = {
            "W1_diag": curr_alpha,
            "W2": curr_alpha
        }

        lr_te_W2 = {
            "W1_diag": -jnp.inf, 
            "W2": curr_alpha
        }

        lr_te_W1 = {
            "W1_diag": curr_alpha, 
            "W2": -jnp.inf
        }
        return lr_gd, lr_te_W2, lr_te_W1

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
    
    @jit 
    def _apply_updates(w, grads, lrs):
        return jax.tree_util.tree_map(lambda p, g, lr: p - jnp.exp(lr) * g, w, grads, lrs)

    def _step(w, data_point):
        x, y, rew, curr_alpha = data_point

        lr_gd, lr_te_W2, lr_te_W1 = _get_lr_pytree(curr_alpha)

        # compute GD gradients -- this will be used in both GD and TE rules
        _, gd_grads = value_and_grad(_squared_error_loss_fn)(w, x, y, rew)
        w_new = None

        # now, do weight updates based on either GD or TE
        if learning_rule == 'gd':
            w_new = _apply_updates(w, gd_grads, lr_gd)
        else:
            _, te_grads_W2 = jax.value_and_grad(_tutor_executor_loss_fn)(w, x, y, rew)
            w_new = _apply_updates(w, te_grads_W2, lr_te_W2)
            w_new = _apply_updates(w_new, gd_grads, lr_te_W1)

        # for the current weights compute the log-likelihood of the data given the hyperparams (alpha, beta)
        out, _ = cortico_striatal_net(w, x)
        ll = _log_likelihood(out, y)

        return w_new, (ll, w_new)

    # inputs_and_outputs_and_rewards = jnp.concatenate((inputs, outputs.reshape(-1, 1), rewards.reshape(-1, 1)), axis=-1)    # num_trials x (input_dim (3) + output_dim (1))
    data_seq = (inputs, outputs, rewards, alpha_arr)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    total_nll = -jnp.sum(lls)

    return total_nll, ws 

batched_run_blocked_deep_forward_weight_trajectory = vmap(run_deep_forward_weight_trajectory, in_axes=(None, 0, 0, 0, None, None, None))




def compute_nll(hyperparams, inputs, outputs, rewards, hyperparams_is_list=True, nln=jax.nn.identity, learning_rule='gd', 
                num_alpha_blocks=1, l2_lambda=1e-3, model_type='two-layer'):
    assert learning_rule in ['gd', 'te']
    assert model_type in ['two-layer', 'one-layer']

    total_nll = None
    if model_type == 'two-layer':
        if num_alpha_blocks == 1:
            total_nll, _ = run_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, learning_rule, hyperparams_is_list, nln)
            # w0 = hyperparams[2:]
            # total_nll + 1e-4 * jnp.sum(w0**2)
        else:
            total_nll, _ = run_blocked_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, learning_rule, hyperparams_is_list, num_blocks=num_alpha_blocks)

            alpha = None
            if hyperparams_is_list:
                alpha = hyperparams[:num_alpha_blocks]
            else:
                alpha = hyperparams['alpha']

            l2_prior = l2_lambda * jnp.sum(jnp.diff(alpha)**2)
            total_nll += l2_prior
    else:
        total_nll, _ = run_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, learning_rule, hyperparams_is_list)

    return total_nll


def compute_batched_nll(hyperparams, inputs, outputs, rewards, hyperparams_is_list=True, nln=jax.nn.identity, learning_rule='gd', 
                        model_type='two-layer'):
    assert learning_rule in ['gd', 'te']
    assert model_type in ['two-layer', 'one-layer']

    all_nlls = None
    if model_type == 'two-layer':
        all_nlls, _ = batched_run_deep_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, learning_rule, hyperparams_is_list, nln)
    else:
        all_nlls, _ = batched_run_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, learning_rule, hyperparams_is_list)

    batch_nll = jnp.mean(all_nlls)

    return batch_nll


def compute_hessian(hyperparams_arr, inputs, outputs, rewards, nln=jax.nn.identity, learning_rule='gd', num_alpha_blocks=1, l2_lambda=1e-3, model_type='two-layer', batched=False):
    assert learning_rule in ['gd', 'te']
    assert model_type in ['two-layer', 'one-layer']
    hess_fn = None 
    hessian = None 
    if batched:
        hess_fn = jax.hessian(compute_batched_nll)
        hessian = hess_fn(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule, model_type=model_type)
    else:
        hess_fn = jax.hessian(compute_nll)
        hessian = hess_fn(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule, num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda, model_type=model_type)
    
    # hessian = jacfwd(jacrev(compute_nll))(jnp_hyperparams, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=True)
    eigvals = jnp.linalg.eigvalsh(hessian)
    return hessian, eigvals 


def optax_fit_params(init_hyperparams, inputs, outputs, rewards, learning_rule='gd', nln=jax.nn.identity,
                     max_iters=500, tol=1e-6, num_alpha_blocks=1, l2_lambda=1e-3, model_type='two-layer', batched=False):
    assert learning_rule in ['gd', 'te']
    assert model_type in ['two-layer', 'one-layer']

    jnp_hyperparams = jnp.concatenate([init_hyperparams[key].reshape(-1) for key in init_hyperparams.keys()])
    meta_loss_fn = None 
    if batched:
        meta_loss_fn = partial(compute_batched_nll, inputs=inputs, outputs=outputs, rewards=rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule, model_type=model_type)
    else:
        meta_loss_fn = partial(compute_nll, inputs=inputs, outputs=outputs, rewards=rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule, 
                           num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda, model_type=model_type)
    opt = optax.lbfgs(memory_size=50) #(memory_size=50)
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


def sophisticated_fit_params(init_hyperparams, inputs, outputs, rewards, learning_rule='gd', 
                             max_iters=2_000, num_alpha_blocks=1, l2_lambda=1e-3, model_type='two-layer', batched=False):
    assert learning_rule in ['gd', 'te']
    assert model_type in ['two-layer', 'one-layer']

    jnp_hyperparams = jnp.concatenate([init_hyperparams[key].reshape(-1) for key in init_hyperparams.keys()])
    
    if batched:
        meta_loss_fn = partial(compute_batched_nll, inputs=inputs, outputs=outputs, rewards=rewards, hyperparams_is_list=True, learning_rule=learning_rule, model_type=model_type)
    else:
        meta_loss_fn = partial(compute_nll, inputs=inputs, outputs=outputs, rewards=rewards, hyperparams_is_list=True, learning_rule=learning_rule, 
                           num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda, model_type=model_type)
        
    x0 = jnp_hyperparams
    N = len(x0)

    solver = BFGS(
        fun=meta_loss_fn,
        tol=1e-6,           
        maxiter=max_iters
    )
    
    res = solver.run(init_params=x0)
    opt_hyperparams = res.params

    return opt_hyperparams



def epsilon_ball_test(hyperparams_arr, inputs, outputs, rewards, key, nln=jax.nn.identity, learning_rule='gd', num_samples=25, epsilon=0.01):
    assert learning_rule in ['gd', 'te']
    N = len(hyperparams_arr)
    curr_nll, _ = run_deep_forward_weight_trajectory(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule)
    nearby_nlls = []
    for _ in range(num_samples):
        _, key = jr.split(key, 2)
        perturbed_hyperparams = hyperparams_arr + jr.normal(key, (N,)) * epsilon 
        inferred_nll, _ = run_deep_forward_weight_trajectory(perturbed_hyperparams, inputs, outputs, rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule)
        nearby_nlls.append(inferred_nll.item())
    return curr_nll, nearby_nlls


def run_randomized_optimization_procedure(inputs, outputs, rewards, key, nln=jax.nn.identity, learning_rule='gd',
                                          num_seeds=25, max_iters=500, tol=1e-4,
                                          alpha_init_scale=1e-1, beta_init_scale=1e0, w0_init_scale=1e-1,
                                          num_alpha_blocks=1, l2_lambda=1e-3, model_type='two-layer', batched=False):
    assert learning_rule in ['gd', 'te']
    assert model_type in ['two-layer', 'one-layer']

    min_nll = None 
    best_w0 = None 
    best_hyperparams_arr = None
    for i in range(num_seeds):
        _, key = jr.split(key, 2)
        subkeys = jr.split(key, 6)

        beta0 = beta_init_scale * jnp.abs(jr.normal(subkeys[1], (1,)).astype(jnp.float32)) #jnp.array([beta_star]) + 5 * jr.normal(subkeys[1], (1,)) 
        alpha_inits = alpha_init_scale * jnp.abs(jr.normal(subkeys[2], (num_alpha_blocks,))) #jnp.array([gd_W1_alpha_star]) + 0.0029 * 2 * (jr.uniform(subkeys[2], (1,)) - 0.5)

        beta0 = jnp.log(beta0)
        alpha_inits = jnp.log(alpha_inits)

        W1_diag_0 = w0_init_scale * jr.normal(subkeys[3], (3,)) #w0['W1_diag'] + 0.1 * jr.normal(subkeys[3], (3, ))  #   # 
        W2_0 = w0_init_scale * jr.normal(subkeys[4], (2, 3)) # w0['W2'] + 0.1 * jr.normal(subkeys[4], (2, 3))   # 

        init_hyperparams = None
        if model_type == 'two-layer':
            init_hyperparams = {'alpha' : alpha_inits, 'beta' : beta0[0], 'W1_diag_0' : W1_diag_0, 'W2_0' : W2_0}
        else:
            init_hyperparams = {'alpha' : alpha_inits, 'beta' : beta0[0], 'W' : W2_0}

        # value = compute_nll(init_hyperparams, inputs, outputs, rewards, hyperparams_is_list=False, learning_rule='gd', num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda)

        hyperparams_arr, _ = optax_fit_params(init_hyperparams, inputs, outputs, rewards, nln=nln, learning_rule=learning_rule, max_iters=max_iters, tol=tol, num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda, model_type=model_type, batched=batched)

        hyperparams = None
        if model_type == 'two-layer':
            hyperparams = {'alpha' : hyperparams_arr[:num_alpha_blocks], 'beta' : hyperparams_arr[num_alpha_blocks], 'W1_diag_0' : hyperparams_arr[num_alpha_blocks+1:num_alpha_blocks+4], 'W2_0' : hyperparams_arr[num_alpha_blocks+4:num_alpha_blocks+10].reshape(2, 3)}
        else:
            hyperparams = {'alpha' : hyperparams_arr[:num_alpha_blocks], 'beta' : hyperparams_arr[num_alpha_blocks], 'W' : hyperparams_arr[num_alpha_blocks+1:num_alpha_blocks+7].reshape(2, 3)}

        inferred_nll = None 
        if batched:
            inferred_nll = compute_batched_nll(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule, model_type=model_type)
        else:
            if model_type == 'two-layer':
                if num_alpha_blocks == 1:
                    inferred_nll, _ = run_deep_forward_weight_trajectory(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule)
                else:
                    inferred_nll, _ = run_blocked_deep_forward_weight_trajectory(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=learning_rule, num_blocks=num_alpha_blocks)
            else:
                inferred_nll, _ = run_onelayer_forward_weight_trajectory(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=learning_rule)

        eigvals = None 
        grad = None
        if not jnp.isnan(inferred_nll):
            _, eigvals = compute_hessian(hyperparams_arr, inputs, outputs, rewards, nln=nln, learning_rule=learning_rule, num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda, model_type=model_type, batched=batched)

            if batched:
                grad = jax.grad(compute_batched_nll)(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule, model_type=model_type)
            else:
                grad = jax.grad(compute_nll)(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, nln=nln, learning_rule=learning_rule, num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda, model_type=model_type)

        # \nNearby NLLs: {nearby_nlls}
        print(f'Iter {i+1} Inferred NLL: {inferred_nll} \nGradient: {grad} \nHessian eigvals: {eigvals}')
        print('\n\n')

        print(f'Inferred alpha: {jnp.exp(hyperparams_arr[0])}, Inferred beta: {jnp.exp(hyperparams_arr[1])}')
        if model_type == 'two-layer':
            print(f'Inferred W1_diag_0: {hyperparams["W1_diag_0"]}, Inferred W2_0: {hyperparams["W2_0"]}')
        else:
            print(f'Inferred W_0: {hyperparams["W"]}')
        print('\n\n')

        if ((min_nll is None and not jnp.isnan(inferred_nll)) or (not jnp.isnan(inferred_nll) and inferred_nll < min_nll)):
            min_nll = inferred_nll
            if model_type == 'two-layer':
                best_w0 = {'W1_diag' : hyperparams['W1_diag_0'], 'W2' : hyperparams['W2_0']}
            else:
                best_w0 = {'W' : hyperparams['W']}
            best_hyperparams_arr = hyperparams_arr

    return best_w0, min_nll, best_hyperparams_arr



def generate_train_test_split(true_alpha, true_beta, true_learning_rule, true_model_type, 
                              session_info, key, nln=jax.nn.identity, num_train_sets=1, num_test_sets=1, vary_inputs=False, 
                              true_w0=None, do_bias_stratified_split=False, one_layer_scale=None,
                              fixed_inputs=None):
    assert true_learning_rule in ['gd', 'te']
    assert true_model_type in ['two-layer', 'one-layer']

    def _show_bias_stats(results_dict, z):
        eff_num_trials = results_dict['eff_num_trials']
        rightward_statistics = results_dict['rightward_statistics']
        partial_rightward_statistics = results_dict['partial_rightward_statistics']

        model_agnostic_bias = calculate_model_agnostic_bias(rightward_statistics, z)
        partial_agnostic_bias = calculate_model_agnostic_bias(partial_rightward_statistics, z)
        print(f'\nModel-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
        print(f'\nModel-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')
        return partial_agnostic_bias, model_agnostic_bias, rightward_statistics

    assert true_learning_rule in ['gd', 'te']
    skeys = jr.split(key, 10)
    (num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast) = session_info

    gd_rate = 0.
    te_rate = 0.

    if true_learning_rule == 'gd':
        gd_rate = true_alpha 
    else:
        te_rate = true_alpha
    
    gd_W1_alpha_star = gd_rate + te_rate
    gd_W2_alpha_star = gd_rate
    te_W2const_alpha_star = te_rate
    te_W2stim_alpha_star = te_rate

    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session

    gd_with_nln = False 
    if nln is not jax.nn.identity:
        gd_with_nln = True

    w0 = None
    if true_w0 is None: 
        if true_model_type == 'two-layer':
            if true_learning_rule == 'gd':
                w0, (_, _) = random_init_params(skeys[1], 3, 3, 2, default_single_loss=True, gd_with_nln=gd_with_nln)
            else:
                w0, (_, _) = random_init_params(skeys[1], 3, 3, 2, default_tutor_executor=True)
        else:
            if one_layer_scale is None:
                w0, (_, _) = shallow_net_params(skeys[1])
            else:
                w0, (_, _) = shallow_net_params(skeys[1], const_scale=one_layer_scale, stim_scale=one_layer_scale)
    else:
        w0 = true_w0

    sskey = skeys[2]

    train_test_split = []

    full_data_dict = {'inputs' : [], 'outputs' : [], 'rewards' : [], 'q_vals' : [],
                      'pma_biases' : [], 'ma_biases' : [], 'rightward_stats' : []}

    train_test_dict = {'train_inputs' : [], 'train_outputs' : [], 'train_rewards' : [], 'train_q_vals' : [],
                       'test_inputs' : [], 'test_outputs' : [], 'test_rewards' : [], 'test_q_vals' : [],
                       'train_pma_biases' : [], 'train_ma_biases' : [], 
                       'test_pma_biases' : [], 'test_ma_biases' : [],
                       'train_rightward_stats' : [], 'test_rightward_stats' : [],
                       'alpha' : true_alpha, 'beta' : true_beta, 'w0' : w0, 'learning_rule' : true_learning_rule,
                       'num_train_sets' : num_train_sets, 'num_test_sets' : num_test_sets}

    # inputs, targets, mask = generate_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, skeys[5])
    inputs, targets, mask = generate_randomized_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, 
                                                                         num_zero_contrast, skeys[5])
    inputs = jnp.array(inputs)
    if fixed_inputs is not None:
        assert vary_inputs == False
        inputs = fixed_inputs
        targets = jnp.argmax(inputs[:, 1:], axis=1)
        targets = targets.reshape(-1, 1)

    for i in range(num_test_sets+num_train_sets):
        _, sskey, ssskey = jr.split(sskey, 3)
        subkeys = jr.split(ssskey, 5)

        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
        w0_copy2 = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
        # inputs, targets, mask = generate_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, sskey)

        if vary_inputs:
            inputs, targets, mask = inputs, targets, mask = generate_randomized_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, 
                                                                                                         num_zero_contrast, subkeys[2])
            inputs = jnp.array(inputs)
        else:
            inputs = jnp.copy(inputs)

        results_dict = None
        if true_model_type == 'two-layer':
            _, _, results_dict = fit_deep_net(w0_copy, num_trials, inputs, targets, None, true_beta, subkeys[4], nln=nln, loss_type='gd_and_te',
                                          end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels,
                                          nonneg_constraint=False, gd_te_lrs=(gd_W1_alpha_star, gd_W2_alpha_star, te_W2const_alpha_star, te_W2stim_alpha_star), num_lr_blocks=1)
        else:
            loss_type = None 
            if true_learning_rule == 'gd':
                loss_type = 'single_loss_gd'
            else:
                loss_type = 'tutor_executor'
            _, _, results_dict = fit_shallow_model(w0_copy, num_trials, inputs, targets, true_alpha, true_beta, subkeys[4], loss_type,
                                                   end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels, nonneg_constraint=False)

        # inputs = jnp.array(inputs)
        outputs = jnp.array(results_dict['outputs'])
        rewards = jnp.array(results_dict['rewards'])
        q_vals = jnp.array(results_dict['q_vals'])

        pma_bias, ma_bias, rightward_stats = _show_bias_stats(results_dict, z)

        full_data_dict['inputs'].append(inputs)
        full_data_dict['outputs'].append(outputs)
        full_data_dict['rewards'].append(rewards)
        full_data_dict['q_vals'].append(q_vals)
        full_data_dict['pma_biases'].append(pma_bias)
        full_data_dict['ma_biases'].append(ma_bias)
        full_data_dict['rightward_stats'].append(rightward_stats)
    
    if do_bias_stratified_split:
        all_ma_biases = np.array(full_data_dict['ma_biases'])
        train_idxs, test_idxs = stratified_quantile_split_indices(all_ma_biases, K=num_train_sets, n_bins=3, random_state=np.random.randint(1_000))
    else:
        train_idxs = np.arange(num_train_sets)
        test_idxs = np.arange(num_train_sets, num_train_sets + num_test_sets)
    
    for idx in train_idxs:
        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
        train_test_split.append((full_data_dict['inputs'][idx], full_data_dict['outputs'][idx], full_data_dict['rewards'][idx], w0_copy))
    for idx in test_idxs:
        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
        train_test_split.append((full_data_dict['inputs'][idx], full_data_dict['outputs'][idx], full_data_dict['rewards'][idx], w0_copy))
    
    train_test_dict['train_inputs'] = [full_data_dict['inputs'][i] for i in train_idxs]
    train_test_dict['train_outputs'] = [full_data_dict['outputs'][i] for i in train_idxs]
    train_test_dict['train_rewards'] = [full_data_dict['rewards'][i] for i in train_idxs]
    train_test_dict['train_q_vals'] = [full_data_dict['q_vals'][i] for i in train_idxs]
    train_test_dict['train_pma_biases'] = [full_data_dict['pma_biases'][i] for i in train_idxs]
    train_test_dict['train_ma_biases'] = [full_data_dict['ma_biases'][i] for i in train_idxs]
    train_test_dict['train_rightward_stats'] = [full_data_dict['rightward_stats'][i] for i in train_idxs]
    
    train_test_dict['test_inputs'] = [full_data_dict['inputs'][i] for i in test_idxs]
    train_test_dict['test_outputs'] = [full_data_dict['outputs'][i] for i in test_idxs]
    train_test_dict['test_rewards'] = [full_data_dict['rewards'][i] for i in test_idxs]
    train_test_dict['test_q_vals'] = [full_data_dict['q_vals'][i] for i in test_idxs]
    train_test_dict['test_pma_biases'] = [full_data_dict['pma_biases'][i] for i in test_idxs]
    train_test_dict['test_ma_biases'] = [full_data_dict['ma_biases'][i] for i in test_idxs]
    train_test_dict['test_rightward_stats'] = [full_data_dict['rightward_stats'][i] for i in test_idxs]
    
    for key in train_test_dict.keys():
        if key in ['train_inputs', 'train_outputs', 'train_rewards', 'train_q_vals', 'test_inputs', 'test_outputs', 'test_rewards', 'test_q_vals']:
            train_test_dict[key] = jnp.asarray(train_test_dict[key])
    
    return train_test_split, train_test_dict


def run_test_set_validation(true_hyperparams, inferred_hyperparams, true_learning_rule, test_learning_rule, test_data, test_model_type='two-layer', nln=jax.nn.identity, plot_trajectories=True):
    assert test_model_type in ['two-layer', 'one-layer']

    (test_inputs, test_outputs, test_rewards) = test_data
    true_nll, true_ws = run_deep_forward_weight_trajectory(true_hyperparams, test_inputs, test_outputs, test_rewards, nln=nln, learning_rule=true_learning_rule, hyperparams_is_list=True)

    if test_model_type == 'two-layer':
        inferred_nll, inferred_ws = run_deep_forward_weight_trajectory(inferred_hyperparams, test_inputs, test_outputs, test_rewards, nln=nln, learning_rule=test_learning_rule, hyperparams_is_list=True)
    else:
        inferred_nll, inferred_ws = run_onelayer_forward_weight_trajectory(inferred_hyperparams, test_inputs, test_outputs, test_rewards, test_learning_rule, hyperparams_is_list=True)

    print(f'True Data NLL (with true learning rule) on test set: {true_nll}')
    print(f'Inferred NLL (with test learning rule) on test set: {inferred_nll}')

    if plot_trajectories:
        true_ws_arr = flatten_params(true_ws)
        inferred_ws_arr = flatten_params(inferred_ws)

        fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)

        realigned_inferred_ws = realign_weight_trajectories(true_ws_arr, inferred_ws_arr)

        i = 0
        for ax in axs.ravel():
            ax.plot(true_ws_arr[:, i], lw=3, color='green', label='True Weight')
            ax.plot(realigned_inferred_ws[:, i], lw=1.8, linestyle='dashed', color='black', alpha=0.8, label='Inferred Weight')
            i += 1
        plt.show()

    return


def run_multi_trial_learning_rule_validation(true_learning_rule, true_model_type, num_trials, key, nln=jax.nn.identity, num_train_per_trial=1, num_test_per_trial=1, 
                                             session_info=(40, 4, 30, 10), num_opt_seeds=20, fixed_alpha=None, fixed_beta=None, vary_inputs=False, 
                                             true_w0=None, do_bias_stratified_split=False, save_data=True, name=''):
    assert true_learning_rule in ['gd', 'te']
    assert true_model_type in ['two-layer', 'one-layer']
    true_num_params = 11
    if true_model_type == 'one-layer':
        true_num_params = 8
    _, key, skey = jr.split(key, 3)

    # true_model_type = 'two-layer'

    print(f'TRUE MODEL: a {true_model_type} network trained with the {true_learning_rule} learning rule.')

    # 5 columns, in order, are true data, 2-layer GD, 2-layer TE, 1-layer GD, 1-layer TE
    model_types = [true_model_type, 'two-layer', 'two-layer', 'one-layer', 'one-layer']
    test_learning_rules = [true_learning_rule, 'gd', 'te', 'gd', 'te']

    test_nlls = np.zeros((num_trials, 5, num_test_per_trial))
    train_nlls = np.zeros((num_trials, 5))
    true_hyperparams_collection = np.zeros((num_trials, true_num_params))
    inferred_hyperparams_collection = np.zeros((num_trials, 4, 11))

    train_pma_biases = np.zeros((num_trials, num_train_per_trial))
    test_pma_biases = np.zeros((num_trials, num_test_per_trial))

    train_rightward_stats = np.zeros((num_trials, num_train_per_trial, 2, 3))
    test_rightward_stats = np.zeros((num_trials, num_test_per_trial, 2, 3))

    alphas = np.zeros((num_trials, 5))
    betas = np.zeros_like(alphas)

    for i in range(num_trials):
        _, skey, sskey = jr.split(skey, 3)
        skeys = jr.split(sskey, 8)
        opt_key = skeys[5]

        alpha_star, beta_star = None, None
        if fixed_alpha is not None:
            assert fixed_beta is not None 
            alpha_star = fixed_alpha
            beta_star = fixed_beta 
        else:
            alpha_star = jr.uniform(skeys[1], minval=0.002, maxval=0.02)
            beta_star = jr.uniform(skeys[7], minval=5.0, maxval=15.0)

        datasets, train_test_dict = generate_train_test_split(alpha_star, beta_star, true_learning_rule, true_model_type, session_info, 
                                                              skeys[3], nln=nln, num_train_sets=num_train_per_trial, num_test_sets=num_test_per_trial, 
                                                              vary_inputs=vary_inputs, true_w0=true_w0, do_bias_stratified_split=do_bias_stratified_split)
        (_, _, _, train_w0) = datasets[0] 
        train_inputs = train_test_dict['train_inputs']
        train_outputs = train_test_dict['train_outputs']
        train_rewards = train_test_dict['train_rewards']

        train_pma_biases[i] = np.asarray(train_test_dict['train_pma_biases'])
        test_pma_biases[i] = np.asarray(train_test_dict['test_pma_biases'])

        train_rightward_stats[i] = np.asarray(train_test_dict['train_rightward_stats'])
        test_rightward_stats[i] = np.asarray(train_test_dict['test_rightward_stats'])

        w0_arr = turn_pytree_to_arr(train_w0)
        alpha_beta_arr = np.log(np.array([alpha_star, beta_star]))
        true_hyperparams_collection[i] = np.concatenate((alpha_beta_arr, w0_arr))


        # if j==0, get nlls coming from true data itself
        for j in range(len(model_types)):
            model_type = model_types[j]
            test_learning_rule = test_learning_rules[j]
            _, opt_key = jr.split(opt_key, 2)

            # TRAINING 
            hyperparams_arr = None 
            if j == 0:
                if true_model_type == 'two-layer':
                    hyperparams_arr = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), train_w0['W1_diag'], train_w0['W2'].reshape(-1)])
                else:
                    hyperparams_arr = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), train_w0['W'].reshape(-1)])
                true_nll = compute_batched_nll(hyperparams_arr, train_inputs, train_outputs, train_rewards, hyperparams_is_list=True, nln=nln, learning_rule=true_learning_rule, model_type=model_type)
                # true_nll, _ = run_deep_forward_weight_trajectory(hyperparams_arr, train_inputs, train_outputs, train_rewards, nln=nln, learning_rule=true_learning_rule, hyperparams_is_list=True)
                train_nlls[i, 0] = true_nll
                print(f'Training Data (ground truth) NLL with {true_learning_rule} as the true learning rule: {true_nll}')
            else:
                print(f'\nPerforming inference in a {model_type} network with the {test_learning_rule} learning rule...')
                num_alpha_blocks = 1
                _, min_nll, hyperparams_arr = run_randomized_optimization_procedure(train_inputs, train_outputs, train_rewards, opt_key, learning_rule=test_learning_rule, model_type=model_type,
                                                                                    num_seeds=num_opt_seeds, max_iters=500, nln=nln, tol=1e-4, 
                                                                                    alpha_init_scale=0.1, beta_init_scale=1., w0_init_scale=0.1, 
                                                                                    num_alpha_blocks=num_alpha_blocks, l2_lambda=0., batched=True)
                # hyperparams = {'alpha' : hyperparams_arr[:num_alpha_blocks], 'beta' : hyperparams_arr[num_alpha_blocks], 'W1_diag_0' : hyperparams_arr[num_alpha_blocks+1:num_alpha_blocks+4], 'W2_0' : hyperparams_arr[num_alpha_blocks+4:num_alpha_blocks+10].reshape(2, 3)}
                train_nlls[i, j] = min_nll
                inferred_hyperparams_collection[i, j-1, :len(hyperparams_arr)] = hyperparams_arr
                print(f'Inferred (training) NLL with {test_learning_rule} as the test learning rule: {min_nll}')

            alphas[i, j] = jnp.exp(hyperparams_arr[0])
            betas[i, j] = jnp.exp(hyperparams_arr[1])

            # TESTING
            if j > 0:
                print(f'\nTesting the {test_learning_rule} learning rule in a {model_type} network...')

            for k in range(num_test_per_trial):
                (test_inputs, test_outputs, test_rewards, _) = datasets[num_train_per_trial + k]
                # test_data = (test_inputs, test_outputs, test_rewards)

                if j == 0:
                    true_test_nll = None 
                    if true_model_type == 'two-layer':
                        true_test_nll, _ = run_deep_forward_weight_trajectory(hyperparams_arr, test_inputs, test_outputs, test_rewards, nln=nln, learning_rule=true_learning_rule, hyperparams_is_list=True)
                    else:
                        true_test_nll, _ = run_onelayer_forward_weight_trajectory(hyperparams_arr, test_inputs, test_outputs, test_rewards, learning_rule=true_learning_rule, hyperparams_is_list=True)
                    test_nlls[i, 0, k] = true_test_nll
                else:
                    inferred_test_nll = None 
                    # len_params = None
                    if model_type == 'two-layer':
                        inferred_test_nll, _ = run_deep_forward_weight_trajectory(hyperparams_arr, test_inputs, test_outputs, test_rewards, test_learning_rule, hyperparams_is_list=True, nln=nln)
                        # len_params = 11
                    else:
                        # len_params = 8
                        inferred_test_nll, _ = run_onelayer_forward_weight_trajectory(hyperparams_arr, test_inputs, test_outputs, test_rewards, test_learning_rule, hyperparams_is_list=True)
                    
                    hyperparams_is_list=True 
                    true_hyperparams = true_hyperparams_collection[i]
                    # inferred_hyperparams = inferred_hyperparams_collection[i, j-1, :len_params]
                    
                    # true_nll, _ = run_deep_forward_weight_trajectory(true_hyperparams, test_inputs, test_outputs, test_rewards, hyperparams_is_list=hyperparams_is_list, learning_rule=true_learning_rule)
                    true_nll = compute_nll(true_hyperparams, test_inputs, test_outputs, test_rewards, hyperparams_is_list=hyperparams_is_list, nln=nln, learning_rule=true_learning_rule, model_type=true_model_type)
                    # true_ws_arr = flatten_params(true_ws)
                    # inferred_ws_arr = flatten_params(inferred_ws)

                    print(f'{model_type} trained with {test_learning_rule}:')
                    print(f'True NLL on test set: {true_nll}')
                    print(f'Inferred NLL on test set: {inferred_test_nll}')
                    print(f'Model-agnostic bias upon reaching expert status: {test_pma_biases[i, k]}')

                    # fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)
                    # plt.title(f'{model_type} trained with {test_learning_rule}')

                    # z = 0
                    # for ax in axs.ravel():
                    #     ax.plot(true_ws_arr[:, z], lw=3, color='green', label='True Weight')
                    #     ax.plot(inferred_ws_arr[:, z], lw=1.8, linestyle='dashed', color='black', alpha=0.8, label='Inferred Weight')
                    #     z += 1
                    # plt.show()
                    
                    test_nlls[i, j, k] = inferred_test_nll
            
            print(f'\nIter {j+1} of 5 completed (within trial {i+1})')
        
        print(f'\nTrial {i+1} of {num_trials} completed')
    
    results_dict = {'train_nlls' : train_nlls, 'test_nlls' : test_nlls, 
                    'true_hyperparams' : true_hyperparams_collection,
                    'inferred_hyperparams' : inferred_hyperparams_collection,
                    'alphas' : alphas, 'betas' : betas,
                    'train_pma_biases' : train_pma_biases, 'test_pma_biases' : test_pma_biases,
                    'train_righward_stats' : train_rightward_stats, 'test_rightward_stats' : test_rightward_stats}
    if save_data:
        filename = f'{name}_simulated_data_inference_true_rule_{true_learning_rule}_numtrials_{num_trials}_numtrainitems{num_train_per_trial}_numtestitems{num_test_per_trial}_numoptseeds{num_opt_seeds}.pkl'
        with open(filename, 'wb') as file:
            pkl.dump(results_dict, file)

    return results_dict




def one_layer_approximation():
    key = jr.PRNGKey(92)   
    _, key, skey = jr.split(key, 3)

    alpha_star = 0.003
    beta_star = 10. 

    true_learning_rule = 'gd'
    true_model_type = 'one-layer'

    num_sessions = 40
    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    session_info = (num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast)

    num_test_sets = 50
    num_train_sets = 2
    vary_inputs = False
    datasets, train_test_dict = generate_train_test_split(alpha_star, beta_star, true_learning_rule, true_model_type, session_info, key, 
                                                          num_train_sets=num_train_sets, num_test_sets=num_test_sets, vary_inputs=vary_inputs)
    
    
    print(train_test_dict['test_pma_biases'])
    print(train_test_dict['test_ma_biases'])
    print('\n')
    (train_inputs, train_outputs, train_rewards, train_w0) = datasets[0] 

    true_hyperparams = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), train_w0['W'].reshape(-1)])

    batched_train_inputs = train_test_dict['train_inputs']
    batched_train_outputs = train_test_dict['train_outputs']
    batched_train_rewards = train_test_dict['train_rewards']


    subkeys = jr.split(skey, 6)

    test_learning_rule = 'gd'
    test_model_type = 'two-layer'
    print(f'\nTesting the {test_learning_rule} learning rule...')
    

    true_batched_nll = compute_batched_nll(true_hyperparams, batched_train_inputs, batched_train_outputs, batched_train_rewards, hyperparams_is_list=True, learning_rule=true_learning_rule, model_type=true_model_type)

    hyperparams_is_list = True
    true_nlls, true_ws = batched_run_onelayer_forward_weight_trajectory(true_hyperparams, batched_train_inputs, batched_train_outputs, batched_train_rewards, true_learning_rule, hyperparams_is_list)

    print(true_nlls)

    print(f'Training Data (ground truth) NLL with {true_learning_rule} as the true learning rule: {true_batched_nll}')

    num_alpha_blocks = 1
    l2_lambda = 0.  # 1.
    hyperparams_is_list = True 
    w0_inferred, min_nll, hyperparams_arr = run_randomized_optimization_procedure(batched_train_inputs, batched_train_outputs, batched_train_rewards, subkeys[5], learning_rule=test_learning_rule,
                                                                                  num_seeds=12, max_iters=1_000, 
                                                                                  tol=1e-6, method='optax',
                                                                                  alpha_init_scale=0.1, beta_init_scale=1., w0_init_scale=0.1, epsilon=0.001,
                                                                                  num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda, batched=True, model_type=test_model_type)
    
    hyperparams = {'alpha' : hyperparams_arr[:num_alpha_blocks], 'beta' : hyperparams_arr[num_alpha_blocks], 'W1_diag_0' : hyperparams_arr[num_alpha_blocks+1:num_alpha_blocks+4], 'W2_0' : hyperparams_arr[num_alpha_blocks+4:num_alpha_blocks+10].reshape(2, 3)}

    # NOW, EVALUATE TRUE MODEL AND TEST MODEL ON TEST DATASETS!

    for k in range(num_test_sets):
        (test_inputs, test_outputs, test_rewards, _) = datasets[num_train_sets + k]
        # test_data = (test_inputs, test_outputs, test_rewards)

        true_test_nll, _ = run_onelayer_forward_weight_trajectory(true_hyperparams, test_inputs, test_outputs, test_rewards, learning_rule=true_learning_rule, hyperparams_is_list=True)
        print(f'\nTrue Model NLL: {true_test_nll}')

        inferred_test_nll, _ = run_deep_forward_weight_trajectory(hyperparams_arr, test_inputs, test_outputs, test_rewards, test_learning_rule, hyperparams_is_list=True)
        print(f'Inferred NLL : {inferred_test_nll}')


    print(f'\nEstimated {test_learning_rule} learning rate(s) for {test_model_type} model: {jnp.exp(hyperparams['alpha'])}')
    print(f'Estimated inverse temperature: {jnp.exp(hyperparams['beta'])}')
    print(f'Estimated init params: {w0_inferred}')
    print(f'\nTrue init params: {train_w0}')
    



def main():
    key = jr.PRNGKey(55)   
    _, key, skey = jr.split(key, 3)

    # _, key, skey, sskey, ssskey, sssskey = jr.split(key, 6)

    # num_sessions = 40

    # expert_threshold = 0.7
    # model_type = 'deep'
    # loss_type = 'gd_and_te'

    # num_grading_levels = 4
    # num_samples_per_grading = 30
    # num_zero_contrast = 10
    # num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    # num_trials = num_sessions * num_samples_per_session
    # z = 1.*num_zero_contrast / num_samples_per_session

    # inputs, targets, mask = generate_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, jr.PRNGKey(11))

    # init_params, (beta_star, alpha_star) = random_init_params(skey, 3, 3, 2, default_single_loss=True)
    # w0 = jax.tree_util.tree_map(lambda x: jnp.array(x), init_params)
    # w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
    # # print(f'True learning rate: {alpha_star}')
    # # print(f'True inverse temperature: {beta_star}')

    # gd_rate = 0.003
    # te_rate = 0.

    # gd_W1_alpha_star = gd_rate + te_rate
    # gd_W2_alpha_star = gd_rate
    # te_W2const_alpha_star = te_rate
    # te_W2stim_alpha_star = te_rate

    # true_learning_rule = 'gd'
    # alpha_star = None 
    # if true_learning_rule == 'gd':
    #     alpha_star = gd_rate
    #     assert te_rate == 0.
    # else:
    #     alpha_star = te_rate
    #     assert gd_rate == 0.

    # beta_star = 10.00

    # params, losses, results_dict = fit_deep_net(init_params, num_trials, inputs, targets, None, beta_star, sskey, loss_type=loss_type,
    #                                             end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels,
    #                                             nonneg_constraint=False, gd_te_lrs=(gd_W1_alpha_star, gd_W2_alpha_star, te_W2const_alpha_star, te_W2stim_alpha_star))

    # outputs = results_dict['outputs']
    # rewards = results_dict['rewards']

    # eff_num_trials = results_dict['eff_num_trials']
    # rightward_statistics = results_dict['rightward_statistics']
    # partial_rightward_statistics = results_dict['partial_rightward_statistics']

    # model_agnostic_bias = calculate_model_agnostic_bias(rightward_statistics, z)
    # partial_agnostic_bias = calculate_model_agnostic_bias(partial_rightward_statistics, z)
    # print(f'\nModel-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
    # print(f'\nModel-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

    # # create copies of the data sequence
    # inputs = jnp.array(inputs, dtype=jnp.float32)
    # outputs = jnp.array(outputs)
    # rewards = jnp.array(rewards, dtype=jnp.float32)

    # print(inputs.shape)
    # print(rewards.shape)
    # print(outputs.shape)

    # accs = calculate_session_accs(rewards, num_samples_per_session, mask)
    # print(f'\nMean Accuracy by Trial: \n{accs}')
    # if accs[:34].max() < expert_threshold:
    #     print('Failed to reach expert status!')
    #     # print(1./0)
    alpha_star = 0.004
    beta_star = 13. 
    true_learning_rule = 'gd'
    true_model_type = 'two-layer'
    nln = jax.lax.erf

    num_sessions = 40
    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    session_info = (num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast)

    num_test_sets = 0
    num_train_sets = 1
    vary_inputs = False
    datasets, train_test_dict = generate_train_test_split(alpha_star, beta_star, true_learning_rule, true_model_type, session_info, key, nln=nln,
                                                          num_train_sets=num_train_sets, num_test_sets=num_test_sets, vary_inputs=vary_inputs)
    
    ground_truth_test_biases = train_test_dict['test_ma_biases']
    
    print(train_test_dict['train_pma_biases'])
    print(train_test_dict['train_ma_biases'])
    print('\n')
    # print(train_test_dict['test_pma_biases'])
    # print(train_test_dict['test_ma_biases'])
    # print('\n')
    (train_inputs, train_outputs, train_rewards, train_w0) = datasets[0] 

    true_hyperparams = None
    if true_model_type == 'two-layer':
        true_hyperparams = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), train_w0['W1_diag'], train_w0['W2'].reshape(-1)])
    else:
        true_hyperparams = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), train_w0['W'].reshape(-1)])

    # w0_arr = turn_pytree_to_arr(train_w0)
    # (train_inputs, train_outputs, train_rewards, train_w0) = train_data 
    # (test_inputs, test_outputs, test_rewards, _) = test_data_with_w0
    # test_data = (test_inputs, test_outputs, test_rewards)
    batched_train_inputs = train_test_dict['train_inputs']
    batched_train_outputs = train_test_dict['train_outputs']
    batched_train_rewards = train_test_dict['train_rewards']


    subkeys = jr.split(skey, 6)

    test_learning_rule = 'gd'
    test_model_type = 'two-layer'
    print(f'\nTesting the {test_learning_rule} learning rule...')
    
    # true_nll, true_ws = run_deep_forward_weight_trajectory(true_hyperparams, train_inputs, train_outputs, train_rewards, hyperparams_is_list=True, learning_rule=true_learning_rule)
    true_batched_nll = compute_batched_nll(true_hyperparams, batched_train_inputs, batched_train_outputs, batched_train_rewards, hyperparams_is_list=True, nln=nln, learning_rule=true_learning_rule, model_type=true_model_type)

    hyperparams_is_list = True
    true_nlls = None 
    if true_model_type == 'two-layer':
        true_nlls, true_ws = batched_run_deep_forward_weight_trajectory(true_hyperparams, batched_train_inputs, batched_train_outputs, batched_train_rewards, true_learning_rule, hyperparams_is_list, nln)
    else:
        true_nlls, true_ws = batched_run_onelayer_forward_weight_trajectory(true_hyperparams, batched_train_inputs, batched_train_outputs, batched_train_rewards, true_learning_rule, hyperparams_is_list)

    print(true_nlls)

    # true_ws_arr = combine_batch_weights(true_ws)    # 2 x T x 9

    # for k in range(num_train_sets):

    #     fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)
        
    #     i = 0
    #     for ax in axs.ravel():
    #         ax.plot(true_ws_arr[k, :, i], lw=2.3, color='green')
    #         i += 1
        
    #     plt.show()




    # true_ws_arr = combine_batch_weights(true_ws)

    print(f'Training Data (ground truth) NLL with {true_learning_rule} as the true learning rule: {true_batched_nll}')

    num_alpha_blocks = 1
    l2_lambda = 0.  # 1.
    hyperparams_is_list = True 
    w0_inferred, min_nll, hyperparams_arr = run_randomized_optimization_procedure(batched_train_inputs, batched_train_outputs, batched_train_rewards, subkeys[5], learning_rule=test_learning_rule,
                                                                                  num_seeds=15, max_iters=500, tol=1e-4, nln=nln,
                                                                                  alpha_init_scale=0.01, beta_init_scale=10., w0_init_scale=0.1,
                                                                                  num_alpha_blocks=num_alpha_blocks, l2_lambda=l2_lambda, batched=True, model_type=test_model_type)
    
    hyperparams = None 
    if test_model_type == 'two-layer':
        hyperparams = {'alpha' : hyperparams_arr[:num_alpha_blocks], 'beta' : hyperparams_arr[num_alpha_blocks], 'W1_diag_0' : hyperparams_arr[num_alpha_blocks+1:num_alpha_blocks+4], 'W2_0' : hyperparams_arr[num_alpha_blocks+4:num_alpha_blocks+10].reshape(2, 3)}
    else:
        hyperparams = {'alpha' : hyperparams_arr[:num_alpha_blocks], 'beta' : hyperparams_arr[num_alpha_blocks], 'W' : hyperparams_arr[num_alpha_blocks+1:num_alpha_blocks+7].reshape(2, 3)}

    inferred_alpha = hyperparams_arr[0]
    inferred_beta = hyperparams_arr[1]


    print(f'Minimum NLL obtained on training data: {min_nll}')
    #hyperparams_arr = jnp.array([hyperparams[key].item() for key in hyperparams.keys()])
    # w0_inferred = {'W1_diag' : hyperparams['W1_diag_0'], 'W2' : hyperparams['W2_0']}

    # print(f'\nTrue GD W1 learning rate: {gd_W1_alpha_star}')
    # print(f'\nTrue GD W2 learning rate: {gd_W2_alpha_star}')

    # print(f'True TE W2const learning rate: {te_W2const_alpha_star}')
    # print(f'True TE W2stim learning rate: {te_W2stim_alpha_star}')
    print(f'True learning rate for {true_learning_rule} rule: {alpha_star}')
    print(f'True inverse temperature: {beta_star}')

    # print(f'Model-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
    # print(f'Model-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

    # print(f'\nInitial learning rates: {alpha_inits * alpha_scale}')
    # print(f'Initial inverse temperature: {beta0}')
    # print(f'Initial starting weights: {W1_diag_0, W2_0}')

    print(f'\nEstimated {test_learning_rule} learning rate(s): {jnp.exp(hyperparams['alpha'])}')
    print(f'Estimated inverse temperature: {jnp.exp(hyperparams['beta'])}')

    print(f'\nTrue init params: {train_w0}')
    print(f'Estimated init params: {w0_inferred}')
    
    # plt.figure()
    # plt.plot(np.arange(n_iters), total_nlls, lw=3, color='blue')
    # plt.xlabel('Iteration of Inference', fontsize=16)
    # plt.ylabel('NLL', fontsize=16)
    # plt.show()
    # plt.savefig('basic_alpha_beta_inference/sample_NLLs_leftbias.png')
    
    # true_hyperparams = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), train_w0['W1_diag'], train_w0['W2'].reshape(-1)])  # {'alpha' : gd_alpha_star / alpha_scale, 'alpha2' : te_alpha_star / alpha_scale, 'beta' : beta_star, 'W1_diag' : init_hyperparams['W1_diag'], 'W2' : init_hyperparams['W2']}
    # inferred_hyperparams = turn_pytree_to_arr(hyperparams) #hyperparams_arr #hyperparams

    # fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)
    # colormap = plt.cm.cool #plt.cm.gist_ncar #nipy_spectral, Set1,Paired
    # colors = [colormap(i) for i in np.linspace(0, 1, len(datasets))]

    # for k in range(len(datasets)):
    #     (curr_inputs, curr_outputs, curr_rewards, _) = datasets[k]
    #     true_nll, true_ws = run_deep_forward_weight_trajectory(true_hyperparams, curr_inputs, curr_outputs, curr_rewards, true_learning_rule, hyperparams_is_list=True)
    #     print(f'{true_nll}')
    #     true_ws_arr = flatten_params(true_ws)

    #     inferred_nll, inferred_ws = run_deep_forward_weight_trajectory(inferred_hyperparams, curr_inputs, curr_outputs, curr_rewards, true_learning_rule, hyperparams_is_list=True)
    #     print(f'{inferred_nll}\n')
    #     inferred_ws_arr = flatten_params(inferred_ws)
    #     realigned_inferred_ws_arr = realign_weight_trajectories(true_ws_arr, inferred_ws_arr)

    #     i = 0
    #     for ax in axs.ravel():
    #         ax.plot(true_ws_arr[:, i], lw=2.3, color=colors[k])
    #         ax.plot(realigned_inferred_ws_arr[:, i], lw=3., color=colors[k], linestyle='dashed')
    #         i += 1
    
    # plt.show()


    for k in range(num_train_sets):

        fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        
        (curr_inputs, curr_outputs, curr_rewards, _) = datasets[k]
        true_nll, true_ws = run_deep_forward_weight_trajectory(true_hyperparams, curr_inputs, curr_outputs, curr_rewards, true_learning_rule, hyperparams_is_list=True, nln=nln)
        print(f'True Data NLL: {true_nll}')

        inferred_nll, inferred_ws = run_deep_forward_weight_trajectory(hyperparams_arr, curr_inputs, curr_outputs, curr_rewards, true_learning_rule, hyperparams_is_list=True, nln=nln)
        print(f'Inferred Test NLL: {inferred_nll}\n')
        # realigned_inferred_ws_arr = realign_weight_trajectories(true_ws_arr, inferred_ws_arr)

        true_eff_trajectories = flatten_params(true_ws)
        true_eff_trajectories = compute_two_layer_flattened_net_params(true_eff_trajectories)

        inferred_eff_trajectories = flatten_params(inferred_ws)
        inferred_eff_trajectories = compute_two_layer_flattened_net_params(inferred_eff_trajectories)

        i = 0
        for ax in axs.ravel():
            ax.plot(true_eff_trajectories[:, i], lw=2.3, color='green', label='True Weights')
            ax.plot(inferred_eff_trajectories[:, i], lw=3., color='black', linestyle='dashed', label='Inferred Weight')
            i += 1
        
        # axs[-1, -1].legend()
        plt.show()
    
    print(1./0)

    
    # NOW SEE HOW BIASED THE INFERRED W0 IS IN THE WEIRD CASES!

    datasets, train_test_dict = generate_train_test_split(alpha_star, beta_star, true_learning_rule, true_model_type, session_info, subkeys[4], 
                                                          num_train_sets=2, num_test_sets=50, vary_inputs=vary_inputs, true_w0=w0_inferred)

    print(train_test_dict)

    new_w0_test_biases = train_test_dict['test_ma_biases']


    datasets, train_test_dict = generate_train_test_split(jnp.exp(inferred_alpha), jnp.exp(inferred_beta), true_learning_rule, true_model_type, session_info, subkeys[3], 
                                                          num_train_sets=2, num_test_sets=50, vary_inputs=vary_inputs, true_w0=train_w0)

    print(train_test_dict)

    test_biases_with_alphabeta_inferred = train_test_dict['test_ma_biases']

    print(jnp.exp(inferred_alpha))
    print(jnp.exp(inferred_beta))
    print(hyperparams_arr)
    print(train_w0)


    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    # Histogram 1
    axes[0].hist(ground_truth_test_biases, bins=10, range=(-1, 1), alpha=0.8, label='ground gruth hyperparams', color='green')
    axes[0].set_title("Ground truth alpha, beta, w0")

    # Histogram 2
    axes[1].hist(new_w0_test_biases, bins=10, range=(-1, 1), alpha=0.8, label='inferring w0', color="blue")
    axes[1].set_title("Inferred w0; ground truth alpha, beta")

    # Histogram 3
    axes[2].hist(test_biases_with_alphabeta_inferred, bins=10, range=(-1, 1), color="red", alpha=0.8, label='inferring alpha, beta')
    axes[2].set_title("Inferred alpha, beta; ground truth w0")

    # Labels
    for ax in axes:
        ax.set_ylabel("Count")
    axes[-1].set_xlabel("Model agnostic bias")
    plt.show()

    # for j in range(num_test_sets):
    #     (test_inputs, test_outputs, test_rewards, _) = datasets[j+1]
    #     test_data = (test_inputs, test_outputs, test_rewards)
    #     run_test_set_validation(true_hyperparams, inferred_hyperparams, true_learning_rule, test_learning_rule, test_data, plot_trajectories=True)




    # # Compare inferred weight trajectories with true ones
    # hyperparams_is_list=True 
    # true_hyperparams = jnp.concatenate([jnp.log(jnp.array([alpha_star, beta_star])), train_w0['W1_diag'], train_w0['W2'].reshape(-1)])  # {'alpha' : gd_alpha_star / alpha_scale, 'alpha2' : te_alpha_star / alpha_scale, 'beta' : beta_star, 'W1_diag' : init_hyperparams['W1_diag'], 'W2' : init_hyperparams['W2']}
    # inferred_hyperparams = turn_pytree_to_arr(hyperparams) #hyperparams_arr #hyperparams

    # true_nll, true_ws = run_deep_forward_weight_trajectory(true_hyperparams, train_inputs, train_outputs, train_rewards, hyperparams_is_list=hyperparams_is_list, learning_rule=true_learning_rule)

    # if num_alpha_blocks == 1:
    #     inferred_nll, inferred_ws = run_deep_forward_weight_trajectory(inferred_hyperparams, train_inputs, train_outputs, train_rewards, hyperparams_is_list=hyperparams_is_list, learning_rule=test_learning_rule)
    # else:
    #     inferred_nll, inferred_ws = run_blocked_deep_forward_weight_trajectory(inferred_hyperparams, train_inputs, train_outputs, train_rewards, hyperparams_is_list=hyperparams_is_list, learning_rule=test_learning_rule, num_blocks=num_alpha_blocks)

    # true_ws_arr = flatten_params(true_ws)
    # inferred_ws_arr = flatten_params(inferred_ws)

    # print(f'True NLL on train set: {true_nll}')
    # print(f'Inferred NLL on train set: {inferred_nll}')

    # # fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)

    # # i = 0
    # # for ax in axs.ravel():
    # #     ax.plot(true_ws_arr[:, i], lw=3, color='green', label='True Weight')
    # #     ax.plot(inferred_ws_arr[:, i], lw=1.8, linestyle='dashed', color='black', alpha=0.8, label='Inferred Weight')
    # #     i += 1
    # # # plt.show()
    # # plt.savefig(f'testing_alternative_rules/og_sample_true_{true_learning_rule}_inferring_{test_learning_rule}.png')
    

    
    # fig, axs = plt.subplots(3, 3, figsize=(18, 10), sharex=True, sharey=True)

    # realigned_inferred_ws = realign_weight_trajectories(true_ws_arr, inferred_ws_arr)

    # i = 0
    # for ax in axs.ravel():
    #     ax.plot(true_ws_arr[:, i], lw=3, color='green', label='True Weight')
    #     ax.plot(realigned_inferred_ws[:, i], lw=1.8, linestyle='dashed', color='black', alpha=0.8, label='Inferred Weight')
    #     i += 1
    # plt.show()

    # plt.savefig(f'testing_alternative_rules/sample_true_{true_learning_rule}_inferring_{test_learning_rule}.png')



def plot_log_alpha_beta_distances(results_dict, trial, true_learning_rule, true_model_type,
                                  savefig=False, filepath=''):
    assert true_learning_rule in ['gd', 'te']
    assert true_model_type in ['two-layer', 'one-layer']
    log_alphas = np.log(results_dict['alphas'])
    log_betas = np.log(results_dict['betas'])

    inferred_alphas = log_alphas[:, 1:]     # num_trials x num_methods
    inferred_betas = log_betas[:, 1:]       # num_trials x num_methods
    num_trials = inferred_alphas.shape[0]

    true_alphas = log_alphas[:, 0].reshape(-1, 1)
    true_betas = log_betas[:, 0].reshape(-1, 1)

    distances = (inferred_alphas - true_alphas)**2 + (inferred_betas - true_betas)**2

    methods = ['2-layer \nGD', '2-layer \nTE', '1-layer \nGD', '1-layer \n TE']

    df = pd.DataFrame({
        'Inference Method': [methods[i] for i in range(4)],
        'Hyperparameter Distance': distances[trial]
    })

    # Plot
    plt.figure(figsize=(4, 4))
    sns.stripplot(data=df, x='Inference Method', y='Hyperparameter Distance', size=10, color='tab:blue')
    plt.title(f"Log Alpha Beta Distance for Trial {trial + 1}")
    # plt.legend()
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'{filepath}/ground_truth_{true_model_type}_{true_learning_rule}_numtrials{num_trials}_trial{trial+1}_log_alpha_beta_distance.png')
    else:
        plt.show()

def plot_train_nlls(results_dict, trial, true_learning_rule, true_model_type,
                    savefig=False, filepath=''):
    assert true_learning_rule in ['gd', 'te']
    assert true_model_type in ['two-layer', 'one-layer']
    train_nlls = results_dict['train_nlls'][trial]
    # Prepare DataFrame
    methods = ['2-layer \nGD', '2-layer \nTE', '1-layer \nGD', '1-layer \n TE']
    df = pd.DataFrame({
        'Method': [methods[i] for i in range(4)],
        'NLL': train_nlls[1:]
    })

    # Plot
    plt.figure(figsize=(5, 4))
    sns.stripplot(data=df, x='Method', y='NLL', size=10, color='tab:blue')
    plt.axhline(train_nlls[0], linestyle='--', color='black', label='True Training Data NLL')
    plt.title(f"Training NLLs Comparison for Trial {trial+1}")
    plt.legend()
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'{filepath}/ground_truth_{true_model_type}_{true_learning_rule}_Trial{trial+1}_training_NLL_comparison.png')
    else:
        plt.show()

def plot_test_nlls(results_dict, trial, true_learning_rule, true_model_type,
                   savefig=False, filepath=''):
    assert true_learning_rule in ['gd', 'te']
    assert true_model_type in ['two-layer', 'one-layer']
    A = results_dict['test_nlls']
    num_test_items = A.shape[-1]

    # Optional method labels (change these to whatever yours are)
    method_labels = {
        1: '2-layer GD',
        2: '2-layer TE',
        3: '1-layer GD',
        4: '1-layer TE'
    }

    # Prepare data for seaborn
    records = []
    for test_set in range(num_test_items):
        baseline_val = A[trial, 0, test_set]  # method 0
        for method in range(1, 5):
            records.append({
                'TestSet': f'Test Set {test_set+1}',
                'Method': method_labels.get(method, f'Method {method}'),
                'NLL': A[trial, method, test_set],
                'Baseline': baseline_val
            })

    df = pd.DataFrame(records)

    # Plot: use stripplot instead of boxplot
    plt.figure(figsize=(13, 5))
    sns.stripplot(data=df, x='TestSet', y='NLL', hue='Method', dodge=True, size=10, palette='tab10', jitter=False)

    # Overlay black bars for ground truth
    # For each test set, draw a horizontal line at the baseline
    xticks = plt.gca().get_xticks()
    test_sets = df['TestSet'].unique()
    for i, test_set in enumerate(test_sets):
        y = df[df['TestSet'] == test_set]['Baseline'].iloc[0]
        plt.hlines(y, i - 0.33, i + 0.33, color='black', linewidth=2.5)

    plt.title(f"Test Set Performance for Trial {trial+1}", fontsize=20)
    plt.ylabel("NLL")
    plt.xlabel("Test Set")
    plt.legend(title='Method')
    plt.tight_layout()

    if savefig:
        plt.savefig(f'{filepath}/ground_truth_{true_model_type}_{true_learning_rule}_Trial{trial+1}_test_NLL_comparison.png')
    else:
        plt.show()



if __name__=="__main__":
    main()
    print(1./0)
    # one_layer_approximation()
    # main()
    # visualize_loss_landscape()

    # true_learning_rule = 'gd'
    # true_model_type = 'one-layer'

    # with open(f'{true_model_type}_{true_learning_rule}_simulated_data_inference_true_rule_{true_learning_rule}_numtrials_3_numtrainitems2_numtestitems5_numoptseeds12.pkl', 'rb') as file:
    #     results_dict = pkl.load(file)
    

    
    true_learning_rule = 'te'
    true_model_type = 'one-layer'
    num_trials = 5
    key = jr.PRNGKey(92)
    _, key = jr.split(key, 2)
    
    results_dict = run_multi_trial_learning_rule_validation(true_learning_rule, true_model_type, num_trials, key, num_train_per_trial=9, 
                                                            num_test_per_trial=9, session_info=(40, 4, 30, 10), num_opt_seeds=15, 
                                                            fixed_alpha=0.003, fixed_beta=10., save_data=True, vary_inputs=False, 
                                                            name=f'stratified_batched_inference/{true_model_type}_{true_learning_rule}/ntrials{num_trials}',
                                                            true_w0=None, do_bias_stratified_split=True)

    filepath = f'stratified_batched_inference/{true_model_type}_{true_learning_rule}'
    # with open(f'{filepath}/ntrials5_simulated_data_inference_true_rule_gd_numtrials_5_numtrainitems9_numtestitems9_numoptseeds15.pkl', 'rb') as file:
    #     results_dict = pkl.load(file)

    # test_pma_biases = results_dict['test_pma_biases']
    # print(test_pma_biases)
    # print(results_dict)

    for i in range(num_trials):
        plot_log_alpha_beta_distances(results_dict, trial=i, true_learning_rule=true_learning_rule, true_model_type=true_model_type, savefig=True, filepath=filepath)

        plot_train_nlls(results_dict, trial=i, true_learning_rule=true_learning_rule, true_model_type=true_model_type, savefig=True, filepath=filepath)

        plot_test_nlls(results_dict, trial=i, true_learning_rule=true_learning_rule, true_model_type=true_model_type, savefig=True, filepath=filepath)

    






