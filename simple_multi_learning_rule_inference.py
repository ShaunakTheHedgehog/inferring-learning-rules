import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax import debug
from jax.tree_util import tree_map
from jax import jacfwd, jacrev
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
from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from utils import generate_multisession_graded_data, calculate_model_agnostic_bias, calculate_session_accs, flatten_params, turn_pytree_to_arr

def run_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, alpha_scale=1., hyperparams_is_list=False):
    """
    Arguments:
    - hyperparams (alphas, beta, w0)
    - inputs
    - outputs
    - rewards

    Returns:
    - total negative log-likelihood of the data given the hyperparams (alphas, beta)
    """

    if hyperparams_is_list:
        alpha_gd_W1 = hyperparams[0]
        alpha_gd_W2 = hyperparams[1]
        alpha_te_W2const = hyperparams[2]
        alpha_te_W2stim = hyperparams[3]
        beta = hyperparams[4]
        W1_diag_0 = hyperparams[5:8]
        W2_0 = hyperparams[8:14].reshape(2, 3)
    else:
        alpha_gd_W1 = hyperparams['alpha_gd_W1']  
        alpha_gd_W2 = hyperparams['alpha_gd_W2']
        alpha_te_W2const = hyperparams['alpha_te_W2const']
        alpha_te_W2stim = hyperparams['alpha_te_W2stim']     
        beta = hyperparams['beta']
        W1_diag_0 = hyperparams['W1_diag_0']
        W2_0 = hyperparams['W2_0']
    
    alphas_te_W2 = jnp.array([alpha_te_W2const, alpha_te_W2stim, alpha_te_W2stim]).reshape(1, -1)
    
    lr_gd = {
        "W1_diag": alpha_gd_W1,
        "W2": alpha_gd_W2
    }

    lr_te = {
        "W1_diag": -jnp.inf, 
        "W2": alphas_te_W2
    }

    w0 = {'W1_diag' : W1_diag_0, 'W2' : W2_0}

    # W1_diag = hyperparams[3:6]
    # W2 = hyperparams[6:15].reshape(2, 3)
    # w0 = {'W1_diag' : W1_diag, 'W2' : W2}

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
        return jax.tree_util.tree_map(lambda p, g, lr: p - jnp.exp(lr) * alpha_scale * g, w, grads, lrs)

    def _step(w, data_point):
        x, y, rew = data_point

        # first, apply the standard single loss GD learning rule to get the next weights w
        _, gd_grads = value_and_grad(_squared_error_loss_fn)(w, x, y, rew)
        _, te_grads_W2 = jax.value_and_grad(_tutor_executor_loss_fn)(w, x, y, rew)
        # w_new = jax.tree_util.tree_map(lambda p, g, alpha_gd: p - alpha_gd * g, w, grads, gd_lrs)

        # w_new = {}
        # w_new["W1_diag"] = w["W1_diag"] - (alpha_gd_W1 * alpha_scale) * gd_grads["W1_diag"]
        # w_new["W2"] = w["W2"] - (alpha_gd_W2 * alpha_scale) * gd_grads["W2"] - (alphas_te_W2 * alpha_scale) * te_grads_W2["W2"]

        w_new = _apply_updates(w, gd_grads, lr_gd)
        w_new = _apply_updates(w_new, te_grads_W2, lr_te)

        # second, for the current weights compute the log-likelihood of the data given the hyperparams (alpha, beta)
        out, _ = cortico_striatal_net(w, x)
        ll = _log_likelihood(out, y)

        return w_new, (ll, w_new)


    # inputs_and_outputs_and_rewards = jnp.concatenate((inputs, outputs.reshape(-1, 1), rewards.reshape(-1, 1)), axis=-1)    # num_trials x (input_dim (3) + output_dim (1))
    data_seq = (inputs, outputs, rewards)
    _, (lls, ws) = lax.scan(_step, w0, data_seq)

    total_nll = -jnp.sum(lls)

    return total_nll, ws 


def compute_nll(hyperparams, inputs, outputs, rewards, alpha_scale=1., hyperparams_is_list=False):
    total_nll, _ = run_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=hyperparams_is_list)
    return total_nll


def optax_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=500, tol=1e-6):
    jnp_hyperparams = jnp.concatenate([init_hyperparams[key].reshape(-1) for key in init_hyperparams.keys()])
    meta_loss_fn = partial(compute_nll, inputs=inputs, outputs=outputs, rewards=rewards, alpha_scale=1., hyperparams_is_list=True)
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


def sophisticated_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=2_000):
    jnp_hyperparams = jnp.concatenate([init_hyperparams[key].reshape(-1) for key in init_hyperparams.keys()])
    meta_loss_fn = partial(compute_nll, inputs=inputs, outputs=outputs, rewards=rewards, alpha_scale=1., hyperparams_is_list=True)
    x0 = jnp_hyperparams
    N = len(x0)

    # solver = ScipyMinimize(
    #     fun=meta_loss_fn,
    #     method="BFGS",
    #     options={"disp": True},
    #     maxiter=max_iters
    # )
    solver = BFGS(
        fun=meta_loss_fn,
        tol=1e-6,           # Tighter function tolerance
        maxiter=max_iters
    )
    res = solver.run(init_params=x0)
    opt_hyperparams = res.params
    return opt_hyperparams


def compute_hessian(hyperparams, inputs, outputs, rewards, alpha_scale=1.):
    jnp_hyperparams = jnp.concatenate([hyperparams[key].reshape(-1) for key in hyperparams.keys()])
    hess_fn = jax.hessian(compute_nll)
    hessian = hess_fn(jnp_hyperparams, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=True)
    # hessian = jacfwd(jacrev(compute_nll))(jnp_hyperparams, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=True)
    eigvals = jnp.linalg.eigvalsh(hessian)
    return hessian, eigvals 


def run_randomized_optimization_procedure(inputs, outputs, rewards, key, 
                                          num_seeds=25, max_iters=500, tol=1e-5,
                                          alpha_init_scale=1e-4, beta_init_scale=1., w0_init_scale=0.1,
                                          method='optax'):
    assert method in ['jaxopt', 'optax']

    min_nll = None 
    best_w0 = None 
    best_hyperparams_arr = None
    for i in range(num_seeds):
        _, key = jr.split(key, 2)
        subkeys = jr.split(key, 6)

        beta0 = beta_init_scale * jnp.abs(jr.normal(subkeys[1], (1,))) #jnp.array([beta_star]) + 5 * jr.normal(subkeys[1], (1,)) 
        alpha_inits = alpha_init_scale * jnp.abs(jr.normal(subkeys[2], (4,))) #jnp.array([gd_W1_alpha_star]) + 0.0029 * 2 * (jr.uniform(subkeys[2], (1,)) - 0.5)

        beta0 = jnp.log(beta0)
        alpha_inits = jnp.log(alpha_inits)

        W1_diag_0 = w0_init_scale * jr.normal(subkeys[3], (3,)) #w0['W1_diag'] + 0.1 * jr.normal(subkeys[3], (3, ))  #   # 
        W2_0 = w0_init_scale * jr.normal(subkeys[4], (2, 3)) # w0['W2'] + 0.1 * jr.normal(subkeys[4], (2, 3))   # 

        init_hyperparams = {'alpha_gd_W1' : alpha_inits[0], 'alpha_gd_W2' : alpha_inits[1], 'alpha_te_W2const' : alpha_inits[2], 'alpha_te_W2stim' : alpha_inits[3], 'beta' : beta0[0], 'W1_diag_0' : W1_diag_0, 'W2_0' : W2_0}

        # hyperparams_arr, _ = optax_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=max_iters, alpha_scale=1., tol=tol)
        hyperparams_arr = None 
        if method == 'jaxopt':
            hyperparams_arr = sophisticated_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=max_iters)
        else:
            hyperparams_arr, _ = optax_fit_params(init_hyperparams, inputs, outputs, rewards, max_iters=max_iters, tol=tol)

        hyperparams = {'alpha_gd_W1' : hyperparams_arr[0], 'alpha_gd_W2' : hyperparams_arr[1], 'alpha_te_W2const' : hyperparams_arr[2], 'alpha_te_W2stim' : hyperparams_arr[3], 'beta' : hyperparams_arr[4], 'W1_diag_0' : hyperparams_arr[5:8], 'W2_0' : hyperparams_arr[8:14].reshape(2, 3)}
        
        inferred_nll, _ = run_forward_weight_trajectory(hyperparams_arr, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=True)

        eigvals = None 
        if not jnp.isnan(inferred_nll):
            _, eigvals = compute_hessian(hyperparams, inputs, outputs, rewards, alpha_scale=alpha_scale)

        print(f'Iter {i+1} Inferred NLL: {inferred_nll}, Hessian eigvals: {eigvals}')

        if ((min_nll is None and not jnp.isnan(inferred_nll)) or (not jnp.isnan(inferred_nll) and inferred_nll < min_nll)):
            min_nll = inferred_nll
            best_w0 = {'W1_diag' : hyperparams['W1_diag_0'], 'W2' : hyperparams['W2_0']}
            best_hyperparams_arr = hyperparams_arr

    return best_w0, min_nll, best_hyperparams_arr


def update_hyperparams(init_hyperparams, w0, inputs, outputs, rewards, n_iters=100, lr=1e-3, alpha_scale=1e-3):

    @jit
    def _update_step(hyperparams):
        total_nll, meta_grads = value_and_grad(compute_nll)(hyperparams, w0, inputs, outputs, rewards, alpha_scale)
        hyperparams = jax.tree_util.tree_map(lambda p, g: jnp.maximum(p - lr * g, 0.), hyperparams, meta_grads)
        return total_nll, hyperparams
    
    total_nlls = []
    curr_hyperparams = init_hyperparams
    for i in range(n_iters):
        total_nll, curr_hyperparams = _update_step(curr_hyperparams)

        if (i+1) % 100 == 0:
            print(f'Iter {i+1}. Total NLL: {total_nll.item()}')
        total_nlls.append(total_nll)

    return curr_hyperparams, total_nlls


if __name__=="__main__":
    key = jr.PRNGKey(341)   
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
    # print(f'True learning rate: {alpha_star}')
    # print(f'True inverse temperature: {beta_star}')

    gd_W1_alpha_star = 0.006
    gd_W2_alpha_star = 0.001
    te_W2const_alpha_star = 0.005
    te_W2stim_alpha_star = 0.005

    beta_star = 8.0

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
    w0 = jax.tree_util.tree_map(lambda x: jnp.array(x), init_params)
    w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
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

    true_hyperparams = jnp.concatenate([jnp.log(jnp.array([gd_W1_alpha_star, gd_W2_alpha_star, te_W2const_alpha_star, te_W2stim_alpha_star, beta_star])), w0['W1_diag'], w0['W2'].reshape(-1)])
    true_nll, true_ws = run_forward_weight_trajectory(true_hyperparams, inputs, outputs, rewards, alpha_scale=1., hyperparams_is_list=True)
    print(true_nll)

    # hyperparams_is_list = True
    # n_iters = 2_000
    # # hyperparams, total_nlls = update_hyperparams(init_hyperparams, w0, inputs, outputs, rewards, n_iters=n_iters, lr=0.001, alpha_scale=alpha_scale)
    # # hyperparams, total_nlls = optax_fit_params(init_hyperparams, w0, inputs, outputs, rewards, n_iters=n_iters, lr=0.1, alpha_scale=alpha_scale)
    # print(init_hyperparams.keys())
    # keys = init_hyperparams.keys()

    # hyperparams_arr = sophisticated_fit_params(init_hyperparams, w0, inputs, outputs, rewards, n_iters=n_iters, alpha_scale=alpha_scale)

    alpha_scale = 1.
    hyperparams_is_list = True 
    w0_inferred, min_nll, hyperparams_arr = run_randomized_optimization_procedure(inputs, outputs, rewards, subkeys[5], 
                                                            num_seeds=15, max_iters=1_000, tol=1e-6, method='jaxopt',
                                                            alpha_init_scale=1e-1, beta_init_scale=1., w0_init_scale=1e-1)

    # hyperparams_arr = jnp.array([hyperparams[key].item() for key in hyperparams.keys()])
    hyperparams = {'alpha_gd_W1' : hyperparams_arr[0], 'alpha_gd_W2' : hyperparams_arr[1], 'alpha_te_W2const' : hyperparams_arr[2], 'alpha_te_W2stim' : hyperparams_arr[3], 'beta' : hyperparams_arr[4], 'W1_diag_0' : hyperparams_arr[5:8], 'W2_0' : hyperparams_arr[8:14].reshape(2, 3)}

    # w0_inferred = {'W1_diag' : hyperparams['W1_diag'], 'W2' : hyperparams['W2']}

    print(f'\nTrue GD W1 learning rate: {gd_W1_alpha_star}')
    print(f'True GD W2 learning rate: {gd_W2_alpha_star}')

    print(f'True TE W2const learning rate: {te_W2const_alpha_star}')
    print(f'True TE W2stim learning rate: {te_W2stim_alpha_star}')

    print(f'True inverse temperature: {beta_star}')

    print(f'Model-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
    print(f'Model-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

    # print(f'\nInitial learning rates: {jnp.exp(alpha_inits) * alpha_scale}')
    # print(f'Initial inverse temperature: {jnp.exp(beta0)}')

    print(f'\nEstimated GD W1 learning rate: {jnp.exp(hyperparams['alpha_gd_W1'])}')
    print(f'Estimated GD W2 learning rate: {jnp.exp(hyperparams['alpha_gd_W2'])}')

    print(f'Estimated TE W2const learning rate: {jnp.exp(hyperparams['alpha_te_W2const'])}')
    print(f'Estimated TE W2stim learning rate: {jnp.exp(hyperparams['alpha_te_W2stim']) }')
    print(f'Estimated inverse temperature: {jnp.exp(hyperparams['beta'])}')

    print(f'\nTrue init params: {init_params}')
    print(f'Estimated init params: {w0_inferred}')
    
    # plt.figure()
    # plt.plot(np.arange(n_iters), total_nlls, lw=3, color='blue')
    # plt.xlabel('Iteration of Inference', fontsize=16)
    # plt.ylabel('NLL', fontsize=16)
    # plt.show()
    # plt.savefig('basic_alpha_beta_inference/sample_NLLs_leftbias.png')


    # Compare inferred weight trajectories with true ones
    true_hyperparams = jnp.concatenate([jnp.log(jnp.array([gd_W1_alpha_star, gd_W2_alpha_star, te_W2const_alpha_star, te_W2stim_alpha_star, beta_star])), w0['W1_diag'], w0['W2'].reshape(-1)])
    print(true_hyperparams.shape)
    inferred_hyperparams = hyperparams_arr #hyperparams

    true_nll, true_ws = run_forward_weight_trajectory(true_hyperparams, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=hyperparams_is_list)

    inferred_nll, inferred_ws = run_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, alpha_scale=alpha_scale, hyperparams_is_list=False)

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










