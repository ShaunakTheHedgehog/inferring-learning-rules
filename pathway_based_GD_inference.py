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

# import simple_rates_and_weights_inference
from simple_rates_and_weights_inference import generate_train_test_split, run_randomized_optimization_procedure
from shallow_linear_net import shallow_net, shallow_net_params, fit_shallow_model
from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from utils import calculate_model_agnostic_bias, calculate_session_accs, flatten_params, turn_pytree_to_arr, realign_weight_trajectories, combine_batch_weights, stratified_quantile_split_indices, compute_two_layer_flattened_net_params, generate_randomized_multisession_graded_data
from metric_GD_network import generate_metric_GD_train_test_split
from metric_GD_inference import run_metric_GD_onelayer_forward_weight_trajectory, batched_run_metric_GD_onelayer_forward_weight_trajectory

def run_deep_pathway_based_GD_forward_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list):
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
        alpha_const = hyperparams[0]
        alpha_stim = hyperparams[1]
        beta = hyperparams[2]
        W1_diag_0 = hyperparams[3:6]
        W2_0 = hyperparams[6:12].reshape(2, 3)
    else:
        alpha_const = hyperparams['alpha_const']
        alpha_stim = hyperparams['alpha_stim']
        beta = hyperparams['beta']
        W1_diag_0 = hyperparams['W1_diag_0']
        W2_0 = hyperparams['W2_0']

    # Use same learning rate (alpha_W2_stim) for both stimulus-related weights
    alphas = jnp.array([alpha_const, alpha_stim, alpha_stim]).reshape(1, -1)

    lrs_gd = {
        "W1_diag": alphas.reshape(-1),
        "W2": alphas
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
batched_run_deep_pathway_based_GD_forward_trajectory = vmap(run_deep_pathway_based_GD_forward_trajectory, in_axes=(None, 0, 0, 0, None))


def run_pathway_based_GD_forward_trajectory(hyperparams, inputs, outputs, rewards, model_type, batched, hyperparams_is_list=True):
    ws = None 
    nll = None 
    if batched:
        if model_type == 'two-layer':
            # pass the final flag positionally so it lines up with vmap(in_axes=(None,0,0,0,None))
            nll, ws = batched_run_deep_pathway_based_GD_forward_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list)
        else:
            nll, ws = batched_run_metric_GD_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list)
    else:
        if model_type == 'two-layer':
            # non-batched call — pass flag positionally for consistency
            nll, ws = run_deep_pathway_based_GD_forward_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list)
        else:
            nll, ws = run_metric_GD_onelayer_forward_weight_trajectory(hyperparams, inputs, outputs, rewards, hyperparams_is_list)

    nll = jnp.mean(nll)

    return nll, ws


def compute_nll(hyperparams, inputs, outputs, rewards, model_type='two-layer', batched=False, hyperparams_is_list=True):
    assert model_type in ['two-layer', 'one-layer']
    nll, _ = run_pathway_based_GD_forward_trajectory(hyperparams, inputs, outputs, rewards, model_type, batched=batched, hyperparams_is_list=hyperparams_is_list)
    return nll


def compute_hessian(hyperparams_arr, inputs, outputs, rewards, model_type='two-layer', batched=False):
    assert model_type in ['two-layer', 'one-layer']

    hess_fn = jax.hessian(compute_nll)
    hessian = hess_fn(hyperparams_arr, inputs, outputs, rewards, model_type, batched, hyperparams_is_list=True)
    
    eigvals = jnp.linalg.eigvalsh(hessian)
    return hessian, eigvals 


def optax_fit_params(init_hyperparams, inputs, outputs, rewards, model_type='two-layer',
                     batched=False, max_iters=500, tol=1e-4):
    assert model_type in ['two-layer', 'one-layer']

    jnp_hyperparams = jnp.concatenate([init_hyperparams[key].reshape(-1) for key in init_hyperparams.keys()])
    meta_loss_fn = partial(compute_nll, inputs=inputs, outputs=outputs, rewards=rewards, model_type=model_type, batched=batched, hyperparams_is_list=True)
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


def run_randomized_pathway_based_GD_inference_procedure(inputs, outputs, rewards, key, num_seeds=25, max_iters=500, tol=1e-4,
                                          alpha_init_scale=1e-1, beta_init_scale=1e0, w0_init_scale=1e-1, model_type='two-layer', batched=False,
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

        alpha_inits = alpha_init_scale * jnp.abs(jr.normal(subkeys[2], (2,))) 
        if alpha_inits_guess is not None:
            alpha_inits = alpha_inits_guess * (1 + noise_level * jr.normal(subkeys[6], (2,)))

        beta0 = jnp.log(beta0)
        alpha_inits = jnp.log(alpha_inits)

        W1_diag_0 = w0_init_scale * jr.normal(subkeys[3], (3,)) 
        W2_0 = w0_init_scale * jr.normal(subkeys[4], (2, 3)) 
        if w0_init_guess is not None:
            W1_diag_0 = w0_init_guess['W1_diag'] * (1 + noise_level * jr.normal(subkeys[7], (3,)))
            W2_0 = w0_init_guess['W2'] * (1 + noise_level * jr.normal(subkeys[8], (2, 3)))

        init_hyperparams = {'alpha_const' : alpha_inits[0], 'alpha_stim' : alpha_inits[1], 
                            'beta' : beta0[0], 'W1_diag_0' : W1_diag_0, 'W2_0' : W2_0}

        hyperparams_arr, _ = optax_fit_params(init_hyperparams, inputs, outputs, rewards, model_type=model_type, 
                                              batched=batched, max_iters=max_iters, tol=tol)

        hyperparams = {'alpha_const' : hyperparams_arr[0], 'alpha_stim' : hyperparams_arr[1], 
                       'beta' : hyperparams_arr[2], 'W1_diag_0' : hyperparams_arr[3:6], 'W2_0' : hyperparams_arr[6:12].reshape(2, 3)}

        inferred_nll = compute_nll(hyperparams_arr, inputs, outputs, rewards, model_type, batched, hyperparams_is_list=True)

        eigvals = None 
        grad = None
        if not jnp.isnan(inferred_nll):
            _, eigvals = compute_hessian(hyperparams_arr, inputs, outputs, rewards, model_type, batched)

            grad = jax.grad(compute_nll)(hyperparams_arr, inputs, outputs, rewards, model_type, batched, hyperparams_is_list=True)

            all_hyperparams.append(hyperparams_arr)
            all_nlls.append(inferred_nll)

        # \nNearby NLLs: {nearby_nlls}
        print(f'Iter {i+1} Inferred NLL: {inferred_nll} \nGradient: {grad} \nHessian eigvals: {eigvals}')
        
        print(f'Inferred alphas: {jnp.exp(hyperparams_arr[0:2])}, Inferred beta: {jnp.exp(hyperparams_arr[2])}')
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

    return best_w0, min_nll, best_hyperparams_arr, (jnp.array(all_hyperparams), jnp.array(all_nlls))


def visualize_inferred_vs_true_effective_weights(inferred_hyperparams_arr, true_hyperparams_arr, inputs, outputs, rewards, 
                                                 true_model_type, test_model_type, plot_both_left_right=True):
    
    true_nll, true_ws = run_pathway_based_GD_forward_trajectory(true_hyperparams_arr, inputs, outputs, rewards, true_model_type, batched=False, hyperparams_is_list=True)

    inferred_nll, inferred_ws = run_pathway_based_GD_forward_trajectory(inferred_hyperparams_arr, inputs, outputs, rewards, test_model_type, batched=False, hyperparams_is_list=True)

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


if __name__ == "__main__":
    key = jr.PRNGKey(50)
    _, key = jr.split(key, 2)
    skeys = jr.split(key, 6)

    true_model_type = 'two-layer'
    test_model_type = 'two-layer'

    num_sessions = 40

    expert_threshold = 0.7

    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session
    session_info = (num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast)

    beta = 10.
    first_lr = 0.
    const_lr = 0.003
    stim_lr = 0.005

    lrs = jnp.array([first_lr, const_lr, stim_lr])


    num_train_sets = 1
    num_test_sets = 1
    datasets, train_test_dict = generate_metric_GD_train_test_split(lrs, beta, 'two-layer', session_info, skeys[3], num_train_sets=num_train_sets, num_test_sets=num_test_sets,
                                                                    vary_inputs=False, true_w0=None, do_bias_stratified_split=False, nonneg_init_weights=True, pathway_based=True,
                                                                    use_saxe_lak_gd_init=True)
    
    (train_inputs, train_outputs, train_rewards, train_w0) = datasets[0]
    batched_train_inputs = train_test_dict['train_inputs']
    batched_train_outputs = train_test_dict['train_outputs']
    batched_train_rewards = train_test_dict['train_rewards']

    ground_truth_test_biases = train_test_dict['test_ma_biases']
    
    print(train_test_dict['train_pma_biases'])
    print(train_test_dict['train_ma_biases'])
    print('\n')


    true_hyperparams_arr = jnp.concatenate([jnp.log(jnp.concatenate([lrs[1:], jnp.array([beta])])), train_w0['W1_diag'], train_w0['W2'].reshape(-1)])
    true_nll = compute_nll(true_hyperparams_arr, batched_train_inputs, batched_train_outputs, batched_train_rewards, true_model_type, batched=True, hyperparams_is_list=True)
    print(f'NLL on true data: {true_nll}')


    w0_inferred, min_nll, inferred_hyperparams_arr, (all_hyperparams, all_nlls) = run_randomized_pathway_based_GD_inference_procedure(batched_train_inputs, batched_train_outputs, batched_train_rewards, skeys[4], num_seeds=15,
                                                                                           max_iters=500, tol=1e-4, alpha_init_scale=0.01, beta_init_scale=10., w0_init_scale=0.1,
                                                                                           model_type=test_model_type, batched=True)
                                                                                           # alpha_inits_guess=lrs[1:], beta_init_guess=beta, w0_init_guess=train_w0, noise_level=0.01)
    
    print(f'Minimum NLL obtained on training data: {min_nll}')
    
    print(f'True learning rates: {lrs[1:]}')

    print(f'\nEstimated learning rate(s): {jnp.exp(inferred_hyperparams_arr[:2])}')
    print(f'Estimated inverse temperature: {jnp.exp(inferred_hyperparams_arr[2])}')

    print(f'\nTrue init params: {train_w0}')
    print(f'Estimated init params: {w0_inferred}')


    for i in range(num_train_sets + num_test_sets):
        (test_inputs, test_outputs, test_rewards, _) = datasets[i]  # num_train_sets+i
        visualize_inferred_vs_true_effective_weights(inferred_hyperparams_arr, true_hyperparams_arr, test_inputs, test_outputs, test_rewards, 
                                                 true_model_type, test_model_type, plot_both_left_right=True)


        visualize_inferred_vs_true_effective_weights(inferred_hyperparams_arr, true_hyperparams_arr, test_inputs, test_outputs, test_rewards, 
                                                 true_model_type, test_model_type, plot_both_left_right=False)


    test_nll = compute_nll(inferred_hyperparams_arr, batched_train_inputs, batched_train_outputs, batched_train_rewards, test_model_type, batched=True, hyperparams_is_list=True)
    print(test_nll)
    print('\n')

    test_hyperparams_arr = jnp.concatenate([jnp.array([jnp.log(0.003), jnp.log(0.005), jnp.log(10.)]), inferred_hyperparams_arr[3:]])
    test_nll = compute_nll(test_hyperparams_arr, batched_train_inputs, batched_train_outputs, batched_train_rewards, test_model_type, batched=True, hyperparams_is_list=True)
    print(f'NLL on hand-picked test hyperparams: {test_nll}')
    

    