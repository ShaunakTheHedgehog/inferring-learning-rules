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
import pickle as pkl
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import scipy.stats as stats

from metric_GD_network import generate_randomized_multisession_graded_data
from inference_utils import *
from inference_pipeline import *
from utils import get_nonlinearity, calculate_running_avg_quantity, calculate_batched_quantity_per_session
from utils import optimal_sign_flip, find_winners, find_model_success_rate
from globals import *


def run_single_model_inference(inference_model, nln_name, depth, num_sessions, key, num_seeds=30, 
                               max_iters=1_000, tol=1e-3, num_gradings=4, num_samples_per_grading=30, 
                               num_zero_contrast_per_session=10, alpha_min=1e-3, alpha_max=1e-2, beta_min=1., 
                               beta_max=10., w0_init_scale=1e-1, layer_two_multiplier=1., ridge_lambda=0.5, 
                               alphabeta_ridge=0.5, mu_alpha=MU_ALPHA, mu_beta=MU_BETA, window=200, stride=50):
   
    nln = get_nonlinearity(nln_name)
    skeys = jr.split(key, 7)

    alpha = jr.uniform(skeys[0], minval=alpha_min, maxval=alpha_max)
    beta = jr.uniform(skeys[1], minval=beta_min, maxval=beta_max)
    w0_arr = generate_init_weights(depth, w0_init_scale, skeys[2], return_array=True, layer_two_multiplier=layer_two_multiplier)
    w0 = convert_weights_to_dict(w0_arr)

    print(f'True alpha: {alpha}, True beta: {beta}')

    inputs, targets, mask = generate_randomized_multisession_graded_data(num_sessions, num_gradings=num_gradings, num_samples_per_grading=num_samples_per_grading, 
                                                                         num_zero_contrast_per_session=num_zero_contrast_per_session, key=skeys[3])
    
    _, _, results_dict = train_gd_network(nln_name, w0, inputs, targets, alpha, beta, skeys[4], num_grading_levels=num_gradings)
    outputs = jnp.array(results_dict['outputs'])
    rewards = jnp.array(results_dict['rewards'])
    ma_bias = results_dict['ma-bias']

    trials = jnp.arange(1, len(rewards)+1)[::stride]
    running_accs = calculate_running_avg_quantity(rewards, window=window, mask=mask)
    running_accs = running_accs[::stride]

    # now, run inference procedure to see if we can recover the true hyperparams
    true_hyperparams_arr = jnp.concatenate([jnp.log(jnp.array([alpha, beta])),  w0_arr.reshape(-1)])
    true_objective = compute_nll(true_hyperparams_arr, inputs, outputs, rewards, nln, inference_model, batched=False, alpha_W1=None, num_weights=DEPTH_TO_NUM_WEIGHTS[depth], ridge_lambda=ridge_lambda, alphabeta_ridge=alphabeta_ridge, mu_alpha=mu_alpha, mu_beta=mu_beta)
    print(f'True hyperparams NLL on training data (with priors): {true_objective}')

    _, _, inferred_hyperparams, optim_dict = run_randomized_optimization_procedure(inputs, outputs, rewards, skeys[5], depth, nln, inference_model, batched=False,
                                                                                    num_seeds=num_seeds, max_iters=max_iters, tol=tol,
                                                                                    alpha_init_scale=1e-2, beta_init_scale=1e1, w0_init_scale=1e-1,
                                                                                    epsilon_seeds=100, epsilon=1e-4, ridge_lambda=ridge_lambda, alphabeta_ridge=alphabeta_ridge,
                                                                                    mu_alpha=mu_alpha, mu_beta=mu_beta, burn_in_iters=max_iters+1)

    nll_no_prior = compute_nll(inferred_hyperparams, inputs, outputs, rewards, nln, inference_model, batched=False, alpha_W1=None, num_weights=DEPTH_TO_NUM_WEIGHTS[depth], ridge_lambda=0., alphabeta_ridge=0., mu_alpha=0., mu_beta=0.)
    aic_no_prior = 2 * nll_no_prior + 2 * len(inferred_hyperparams)

    true_nll_no_prior = compute_nll(true_hyperparams_arr, inputs, outputs, rewards, nln, inference_model, batched=False, alpha_W1=None, num_weights=DEPTH_TO_NUM_WEIGHTS[depth], ridge_lambda=0., alphabeta_ridge=0., mu_alpha=0., mu_beta=0.)
    true_aic = 2 * true_nll_no_prior + 2 * len(true_hyperparams_arr)
    
    print(f'True AIC: {true_aic}')  
    print(f'Inferred Model AIC: {aic_no_prior}')

    inferred_alpha = jnp.exp(inferred_hyperparams[0])
    inferred_beta = jnp.exp(inferred_hyperparams[1])

    alpha_beta_error = (inferred_alpha - alpha)**2 + (inferred_beta - beta)**2
    log_alpha_beta_error = jnp.sum((inferred_hyperparams[:2] - true_hyperparams_arr[:2])**2)

    results_dict = {'inputs': inputs, 'targets': targets, 'outputs': outputs, 'rewards': rewards,
                    'mask': mask, 'trials': trials, 'running_accs': running_accs, 'ma-bias': ma_bias,
                    'true_hyperparams': true_hyperparams_arr, 'model_aic': aic_no_prior, 'model_nll': nll_no_prior, 
                    'true_aic': true_aic, 'true_nll': true_nll_no_prior, 'inferred_hyperparams': inferred_hyperparams,
                    'inferred_alpha': inferred_alpha, 'inferred_beta': inferred_beta,
                    'rel_aics': aic_no_prior - true_aic, 'rel_nlls': nll_no_prior - true_nll_no_prior, 'optim_dict': optim_dict,
                    'alpha_beta_error': alpha_beta_error, 'log_alpha_beta_error': log_alpha_beta_error}


    return results_dict



def run_multi_model_inference(true_model_name, num_sessions, key, num_seeds=30, max_iters=1_000, tol=1e-3,
                              num_gradings=4, num_samples_per_grading=30, num_zero_contrast_per_session=10, alpha_min=1e-3, alpha_max=1e-2,
                              beta_min=1., beta_max=10., w0_init_scale=1e-1, layer_two_multiplier=1.,
                              ridge_lambda=0.5, alphabeta_ridge=0.5, mu_alpha=MU_ALPHA, mu_beta=MU_BETA,
                              window=200, stride=50):
    inference_model, depth, nln_name, _, _ = MODEL_CLASS_INFO[true_model_name]
    nln = get_nonlinearity(nln_name)

    skeys = jr.split(key, 7)

    alpha = jr.uniform(skeys[0], minval=alpha_min, maxval=alpha_max)
    beta = jr.uniform(skeys[1], minval=beta_min, maxval=beta_max)
    w0_arr = generate_init_weights(depth, w0_init_scale, skeys[2], return_array=True, layer_two_multiplier=layer_two_multiplier)
    w0 = convert_weights_to_dict(w0_arr)

    print(f'True alpha: {alpha}, True beta: {beta}')

    inputs, targets, mask = generate_randomized_multisession_graded_data(num_sessions, num_gradings=num_gradings, num_samples_per_grading=num_samples_per_grading, 
                                                                         num_zero_contrast_per_session=num_zero_contrast_per_session, key=skeys[3])
    
    _, _, results_dict = train_gd_network(nln_name, w0, inputs, targets, alpha, beta, skeys[4], num_grading_levels=num_gradings)
    outputs = jnp.array(results_dict['outputs'])
    rewards = jnp.array(results_dict['rewards'])
    ma_bias = results_dict['ma-bias']

    trials = jnp.arange(1, len(rewards)+1)[::stride]
    running_accs = calculate_running_avg_quantity(rewards, window=window, mask=mask)
    running_accs = running_accs[::stride]

    # now, run inference procedure to see if we can recover the true hyperparams
    true_hyperparams_arr = jnp.concatenate([jnp.log(jnp.array([alpha, beta])),  w0_arr.reshape(-1)])
    true_objective = compute_nll(true_hyperparams_arr, inputs, outputs, rewards, nln, inference_model, batched=False, alpha_W1=None, num_weights=DEPTH_TO_NUM_WEIGHTS[depth], ridge_lambda=ridge_lambda, alphabeta_ridge=alphabeta_ridge, mu_alpha=mu_alpha, mu_beta=mu_beta)
    print(f'True hyperparams NLL on training data (with priors): {true_objective}')

    sskey = skeys[6]
    aics = np.zeros(len(MODEL_CLASS_INFO))
    nlls = np.zeros(len(MODEL_CLASS_INFO))
    all_hyperparams = {}

    # Now, we will run the optimization procedure for each of the three model classes, and return the AIC data for each model
    for i, model_name in enumerate(MODEL_CLASS_INFO.keys()):
        _, sskey = jr.split(sskey)
        print(f'Running inference procedure for model class: {model_name}')
        curr_inference_model, curr_depth, curr_nln_name, _, _ = MODEL_CLASS_INFO[model_name]
        curr_nln = get_nonlinearity(curr_nln_name)

        _, _, inferred_hyperparams, _ = run_randomized_optimization_procedure(inputs, outputs, rewards, sskey, curr_depth, curr_nln, curr_inference_model, batched=False,
                                                                                           num_seeds=num_seeds, max_iters=max_iters, tol=tol,
                                                                                           alpha_init_scale=1e-2, beta_init_scale=1e1, w0_init_scale=1e-1,
                                                                                           epsilon_seeds=100, epsilon=1e-4, ridge_lambda=ridge_lambda, alphabeta_ridge=alphabeta_ridge,
                                                                                           mu_alpha=mu_alpha, mu_beta=mu_beta, burn_in_iters=max_iters+1)
        
        nll_no_prior = compute_nll(inferred_hyperparams, inputs, outputs, rewards, curr_nln, curr_inference_model, batched=False, alpha_W1=None, num_weights=DEPTH_TO_NUM_WEIGHTS[curr_depth], ridge_lambda=0., alphabeta_ridge=0., mu_alpha=0., mu_beta=0.)
        nlls[i] = nll_no_prior
        aic = 2 * nll_no_prior + 2 * len(inferred_hyperparams)  # AIC = 2k - 2ln(L), where k is number of parameters and L is likelihood
        aics[i] = aic
        all_hyperparams[model_name] = inferred_hyperparams

    true_nll_no_prior = compute_nll(true_hyperparams_arr, inputs, outputs, rewards, nln, inference_model, batched=False, alpha_W1=None, num_weights=DEPTH_TO_NUM_WEIGHTS[depth], ridge_lambda=0., alphabeta_ridge=0., mu_alpha=0., mu_beta=0.)
    true_aic = 2 * true_nll_no_prior + 2 * len(true_hyperparams_arr)
    print(f'True AIC: {true_aic}')  
    print(f'AICs for each model class: {aics}\n')

    results_dict = {'inputs': inputs, 'targets': targets, 'outputs': outputs, 'rewards': rewards,
                    'mask': mask, 'trials': trials, 'running_accs': running_accs, 'ma-bias': ma_bias,
                    'true_hyperparams': true_hyperparams_arr, 'aics': aics, 'nlls': nlls, 
                    'true_aic': true_aic, 'true_nll': true_nll_no_prior, 'all_inferred_hyperparams': all_hyperparams,
                    'rel_aics': aics - true_aic, 'rel_nlls': nlls - true_nll_no_prior}


    return results_dict



def run_multi_trial_multi_model_inference(ntrials, true_model_name, num_sessions, key, num_seeds=25, max_iters=1_000, tol=1e-3,
                                          num_gradings=4, num_samples_per_grading=30, num_zero_contrast_per_session=10, alpha_min=1e-3, alpha_max=1e-2,
                                          beta_min=1., beta_max=10., w0_init_scale=1e-1, ridge_lambda=0.5, alphabeta_ridge=0.5, mu_alpha=MU_ALPHA, mu_beta=MU_BETA):
    all_aics = []
    relative_aics = []
    for trial in range(ntrials):
        print(f'Running trial {trial+1}/{ntrials}')
        _, key = jr.split(key)
        aics, true_aic, nlls, true_nll = run_multi_model_inference(true_model_name, num_sessions, key, num_seeds, max_iters, tol,
                                        num_gradings, num_samples_per_grading, num_zero_contrast_per_session, alpha_min, alpha_max,
                                        beta_min, beta_max, w0_init_scale, ridge_lambda, alphabeta_ridge, mu_alpha, mu_beta)
        all_aics.append(aics)
        relative_aics.append(aics - true_aic)  # Store the difference between each model's AIC and the true model's AIC for this trial
    all_aics = jnp.array(all_aics)
    relative_aics = jnp.array(relative_aics)
    print(f'All AICs across trials: {all_aics}')
    print(f'Relative AICs across trials: {relative_aics}')
    return all_aics, relative_aics



def plot_recovery(trial_dict, title, ax, color, trials_in_session=250):
    num_sessions_list = sorted(trial_dict.keys())
    means = [trial_dict[n].mean() for n in num_sessions_list]
    sems  = [trial_dict[n].std() / np.sqrt(len(trial_dict[n])) for n in num_sessions_list]
    print(f'Means: {means}')

    num_trials_list = [trials_in_session * n for n in num_sessions_list]  # Convert number of sessions to number of trials for x-axis

    ax.plot(num_trials_list, means, color=color, lw=1.8, marker='o', ms=4, zorder=3)
    ax.fill_between(num_trials_list,
                    np.array(means) - np.array(sems),
                    np.array(means) + np.array(sems),
                    color=color, alpha=0.2, linewidth=0)

    max_pts = np.array(means) + np.array(sems)
    upper_y = np.max(max_pts) + 0.1

    # ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=16)

    # ax.set_xticklabels(num_sessions_list)
    ax.set_xscale('log', base=2)
    ax.set_xticks(num_trials_list)
    ax.set_xticklabels([f'{n:,}' for n in num_trials_list], fontsize=13)  # e.g. "2,000"
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('Number of Trials', fontsize=15)
    # ax.set_ylim(bottom=0.)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=13, length=3, width=0.8)


def calculate_recovery_error(true_hyperparams, inferred_hyperparams, recovery_type, depth,
                             inputs=None, outputs=None, rewards=None, nln_name=None, inference_model='gd'):
    assert recovery_type in ['alpha', 'beta', 'alpha_beta', 'w0', 'full']

    nln = get_nonlinearity(nln_name)

    if recovery_type == 'alpha':
        return (inferred_hyperparams[0] - true_hyperparams[0])**2
    elif recovery_type == 'beta': 
        return (inferred_hyperparams[1] - true_hyperparams[1])**2
    elif recovery_type == 'alpha_beta':
        return jnp.sum((inferred_hyperparams[:2] - true_hyperparams[:2])**2)
    
    # recovery_type == 'w0' or 'full'
    _, true_ws, _ = run_single_learning_forward_trajectory(true_hyperparams, inputs, outputs, rewards, nln, inference_model, alpha_W1=None)
    _, inferred_ws, _ = run_single_learning_forward_trajectory(inferred_hyperparams, inputs, outputs, rewards, nln, inference_model, alpha_W1=None)
    
    true_ws = flatten_params(true_ws)
    inferred_ws = flatten_params(inferred_ws)

    nrows = true_ws.shape[-1] // 3
    true_ws = true_ws.reshape(-1, nrows, 3)
    inferred_ws = inferred_ws.reshape(-1, nrows, 3)

    s = jnp.array([1., 1., 1.])
    # only in 2-layer network do we want to account for the sign flip degeneracy
    if depth == 'two-layer':
        s, _ = optimal_sign_flip(true_ws, inferred_ws)

    # calculate error in init weights, accounting for sign flip degeneracy
    error = jnp.sum((true_ws[0] - inferred_ws[0] * s.reshape(1, 3))**2)

    if recovery_type == 'w0':
        return error

    # recovery_type is now 'full'
    return error + jnp.sum((inferred_hyperparams[:2] - true_hyperparams[:2])**2)


def get_recovery_title(recovery_type):
    assert recovery_type in ['alpha', 'beta', 'alpha_beta', 'w0', 'full']
    title = None
    if recovery_type == 'alpha':
        title = r'Recovery Error for $\alpha$ (Log-space)'
    elif recovery_type == 'beta':
        title = r'Recovery Error for $\beta$ (Log-space)'
    elif recovery_type == 'alpha_beta':
        title = r'Recovery Error for $\alpha$ and $\beta$ (Log-space)'
    elif recovery_type == 'w0':
        title = r'Recovery Error for $w_0$'
    else:
        title = 'Recovery Error for Inferred Hyperparameters'
    return title


def visualize_recovery(inference_model, nln_name, depth, ax, color, 
                                  num_runs=100, num_sessions_list=[2, 4, 8, 16, 32, 64, 128], 
                                  trials_in_session=250, recovery_type='alpha'):
    assert recovery_type in ['alpha', 'beta', 'alpha_beta', 'w0', 'full']
    trial_dict = {}
    for num_sessions in num_sessions_list:
        errors = []
        for i in range(num_runs):
            run_results = pkl.load(open(f'synthetic_data_GD_analysis/single_model/{inference_model}_{nln_name}_{depth}/{depth}_{inference_model}_{nln_name}_model_{num_sessions}_sessions_{i}_single_model_validation.pkl', 'rb'))
            log_alpha_beta_error = run_results['log_alpha_beta_error']
            # log_ab_errors.append(log_alpha_beta_error)
            true_x0 = run_results['true_hyperparams']
            inferred_x0 = run_results['inferred_hyperparams']

            error = calculate_recovery_error(true_x0, inferred_x0, recovery_type, depth=depth,
                                             inputs=run_results['inputs'], outputs=run_results['outputs'], rewards=run_results['rewards'], nln_name=nln_name)
            
            if recovery_type == 'alpha_beta':
                error = log_alpha_beta_error

            errors.append(error)
        
        trial_dict[num_sessions] = np.asarray(errors)

    title = get_recovery_title(recovery_type)
    plot_recovery(trial_dict, title, ax, color, trials_in_session=trials_in_session)

    return

def visualize_recovery_all_models(recovery_type, num_runs=100, num_sessions_list=[2, 4, 8, 16, 32, 64, 128], trials_in_session=250, save_path=None):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for key, (inference_model, depth, nln_name, _, color) in MODEL_CLASS_INFO.items():
        visualize_recovery(inference_model, nln_name, depth, ax, color, num_runs=num_runs, num_sessions_list=num_sessions_list, trials_in_session=trials_in_session, recovery_type=recovery_type)

    ax.set_ylim(bottom=0.)

    legend_elements = [Line2D([0], [0], color=MODEL_CLASS_INFO[m][4], lw=1.8, marker='o', ms=4, label=m) for m in MODEL_CLASS_INFO]
    ax.legend(handles=legend_elements, fontsize=11.5, frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return



def visualize_inferred_weights_and_true_weights(true_model_name, model_idx, save_path=None, stride=1):
    inference_model, depth, nln_name, _, _ = MODEL_CLASS_INFO[true_model_name]
    # true_model_type = f'{depth} {nln_name} {inference_model}'
    results_dict = pkl.load(open(f'synthetic_data_GD_analysis/model_validation_experiments/{inference_model}_{nln_name}_{depth}/{depth}_{inference_model}_{nln_name}_model_{model_idx}_multi_model_validation.pkl', 'rb'))

    inputs = results_dict['inputs']
    outputs = results_dict['outputs'].astype(jnp.int32)
    rewards = results_dict['rewards']

    trials = np.arange(1, len(inputs)+1)

    ma_bias = results_dict['ma-bias']
    print(f'M-A bias for this model: {ma_bias}')

    running_accs = results_dict['running_accs']
    print(f'Final running accuracies for this model: {running_accs[-10:]}')

    true_hyperparams_arr = results_dict['true_hyperparams']
    all_inferred_hyperparams = results_dict['all_inferred_hyperparams']
    true_color = 'black'
    model_colors = [MODEL_CLASS_INFO[model_name][4] for model_name in MODEL_CLASS_INFO.keys()]
    colors = [true_color] + model_colors
    all_hyperparams = [true_hyperparams_arr] + [all_inferred_hyperparams[model_name] for model_name in OLDER_MODEL_NAMES]
    model_names = [f'Ground Truth ({true_model_name})'] + list(MODEL_CLASS_INFO.keys())
    nln_names = [nln_name] + [MODEL_CLASS_INFO[model_name][2] for model_name in MODEL_CLASS_INFO.keys()]
    linestyles = ['dashed'] + ['solid'] * len(MODEL_CLASS_INFO)
    lws = [2.5] + [1.8] * len(MODEL_CLASS_INFO)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True)
    fig.suptitle('Input Loading Differences of Inferred Models', fontsize=17, y=0.99)

    col_titles = ['constant channel', 'right channel', 'left channel']  
    for j, title in enumerate(col_titles):
        ax[j].set_title(title, fontsize=15)

    for i, (hyperparams, color, model_name, nln, linestyle, lw) in enumerate(zip(all_hyperparams, colors, model_names, nln_names, linestyles, lws)):
        nln_fn = get_nonlinearity(nln)
        _, ws, _ = run_single_learning_forward_trajectory(hyperparams, inputs, outputs, rewards,
                                                          nln_fn, inference_model, alpha_W1=None)
        ws_flat = flatten_params(ws)
        loading_stats = gather_input_contributions(ws_flat, nln_fn)
        right_loading_diffs = loading_stats['right_diffs']
        left_loading_diffs = loading_stats['left_diffs']

        ax[0].plot(trials[::stride], right_loading_diffs[:, 0][::stride], color=color, lw=lw, linestyle=linestyle, label=model_name)    # constant channel Q-val diffs
        ax[1].plot(trials[::stride], right_loading_diffs[:, 1][::stride], color=color, lw=lw, linestyle=linestyle, label=model_name)    # right channel Q-val diffs
        ax[2].plot(trials[::stride], left_loading_diffs[:, 2][::stride], color=color, lw=lw, linestyle=linestyle)                # left channel Q-val diffs

    # Shared zero-line and spine cleanup across all axes
    for a in ax.flat:
        a.axhline(0, lw=0.9, color='gray', linestyle=':', zorder=0)
        a.spines[['top', 'right']].set_visible(False)
        a.spines[['left', 'bottom']].set_linewidth(0.8)
        a.tick_params(labelsize=12, length=3, width=0.8)
        a.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
        a.set_axisbelow(True)

    Q = r'$Q_R - Q_L$'

    ax[0].set_ylabel(f'{Q} \nContribution', fontsize=15)
    for j in range(3):
        ax[j].set_xlabel('Trial', fontsize=15)

    legend_elements = [
        Line2D([0], [0], color='black', lw=1.8, linestyle='dashed', label=f'Ground Truth ({true_model_name})'),
        Line2D([0], [0], color=MODEL_CLASS_INFO['1-Layer Linear'][4],  lw=1.8, linestyle='solid', label='1-Layer Linear'),
        Line2D([0], [0], color=MODEL_CLASS_INFO['2-Layer Linear'][4],  lw=1.8, linestyle='solid', label='2-Layer Linear'),
        Line2D([0], [0], color=MODEL_CLASS_INFO['2-Layer Nonlinear'][4],     lw=1.8, linestyle='solid', label='2-Layer Nonlinear'),
    ]
    fig.legend(
        handles=legend_elements,
        fontsize=14, frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.85])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

    return


def run_model_aic_analysis(inference_model, nln_name, depth, num_runs=50, save_path=None, title=None, no_ridge=False):
    all_rel_aics = []
    idxs = np.arange(num_runs)
    no_ridge_label = ''
    sessions_label = ''
    if no_ridge:
        no_ridge_label = 'no_ridge/'
        sessions_label = '_40_sessions'
    for i in idxs:
        results_dict = pkl.load(open(f'synthetic_data_GD_analysis/model_validation_experiments/{no_ridge_label}{inference_model}_{nln_name}_{depth}/{depth}_{inference_model}_{nln_name}_model{sessions_label}_{i}_multi_model_validation.pkl', 'rb'))

        rel_aics = results_dict['rel_aics']
        all_rel_aics.append(rel_aics)
    
    all_rel_aics = np.array(all_rel_aics)   # shape: (num_runs, num_models)

    num_under_zero = np.sum((all_rel_aics < 0), axis=0)
    print(f'Number under zero: {num_under_zero}')

    mean_rel_aics = np.mean(all_rel_aics, axis=0)
    std_rel_aics = np.std(all_rel_aics, axis=0)
    print(f'Mean relative AICs across {num_runs} runs: {mean_rel_aics}')
    print(f'Standard deviation of relative AICs across {num_runs} runs: {std_rel_aics}')

    model_names = list(MODEL_CLASS_INFO.keys())
    colors = [MODEL_CLASS_INFO[model_name][4] for model_name in model_names]
    # title = f'True Model: {depth} {nln_name} {inference_model}'
    fig = raincloud_plot(
        all_rel_aics, model_names, colors,
        title=title,
        save_path=save_path,   # e.g. "figures/aic_comparison.pdf"
    )
    plt.show()


def raincloud_plot(
    all_rel_aics: np.ndarray,        # shape (num_runs, num_models)
    model_names: list[str],
    colors: list[str],
    title: str = "",
    ylabel: str = "Relative AIC",
    figsize: tuple = (7, 5),
    jitter_width: float = 0.06,
    violin_width: float = 0.35,
    point_alpha: float = 0.55,
    point_size: float = 3.5,
    seed: int = 0,
    save_path: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Raincloud plot for relative AIC scores.

    Layout per model (left → right):
      [half-violin]  [box]  [jittered strip]

    Parameters
    ----------
    all_rel_aics : (num_runs, num_models) array
    model_names  : list of model class labels
    colors       : list of hex/named colors, one per model
    title        : figure suptitle (e.g. "True Model: deep ReLU GD")
    save_path    : if given, saves figure to this path
    """
    rng = np.random.default_rng(seed)
    n_models = len(model_names)
    xs = np.arange(n_models)          # integer positions on x-axis

    # ── Figure / axes ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, color) in enumerate(zip(model_names, colors)):
        data = all_rel_aics[:, i]
        x = xs[i]

        # ── 1. Half-violin (kernel density, left side) ────────────────────
        kde = stats.gaussian_kde(data, bw_method="scott")
        y_range = np.linspace(data.min(), data.max(), 300)
        density = kde(y_range)
        density = density / density.max() * violin_width   # normalize width

        ax.fill_betweenx(
            y_range,
            x - density,            # left edge
            x,                      # centre line
            alpha=0.75,
            color=color,
            linewidth=0,
        )
        ax.plot(
            x - density, y_range,
            color=color, linewidth=0.8, alpha=0.9,
        )

        # ── 2. Thin box plot (right side, no fliers) ─────────────────────
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        iqr = q75 - q25
        whisker_lo = max(data.min(), q25 - 1.5 * iqr)
        whisker_hi = min(data.max(), q75 + 1.5 * iqr)
        box_x = x + 0.06
        box_w = 0.10

        # IQR box
        ax.add_patch(FancyBboxPatch(
            (box_x - box_w / 2, q25), box_w, iqr,
            boxstyle="square,pad=0",
            facecolor="white", edgecolor=color, linewidth=1.4, zorder=3,
        ))
        # Median line
        ax.hlines(q50, box_x - box_w / 2, box_x + box_w / 2,
                  colors=color, linewidth=2.2, zorder=4)
        # Whiskers
        ax.vlines(box_x, whisker_lo, q25, colors=color, linewidth=1.1, zorder=3)
        ax.vlines(box_x, q75, whisker_hi, colors=color, linewidth=1.1, zorder=3)
        ax.hlines([whisker_lo, whisker_hi],
                  box_x - box_w / 3, box_x + box_w / 3,
                  colors=color, linewidth=1.1, zorder=3)

        # ── 3. Jittered strip (right of box) ─────────────────────────────
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(data))
        strip_x = x + 0.28 + jitter
        ax.scatter(
            strip_x, data,
            color=color, alpha=point_alpha, s=point_size ** 2,
            linewidths=0, zorder=2,
        )

    # ── Reference line at 0 ──────────────────────────────────────────────────
    ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--",
               dashes=(4, 3), zorder=1, label="Reference (0)")

    # ── Axes styling ─────────────────────────────────────────────────────────
    ax.set_xticks(xs)
    ax.set_xticklabels(['', '', ''], fontsize=10)
    ax.locator_params(axis='y', nbins=4)
    # ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_xlim(-0.6, n_models - 0.4)

    # Clean spines
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=23,
                   length=3, width=0.8)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6,
                  color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # ── Legend for raincloud elements ────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#aaaaaa", alpha=0.7, label="Density (KDE)"),
        mpatches.Patch(facecolor="white",   edgecolor="#555555",
                       linewidth=1.2,       label="IQR box  |  median"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=4, alpha=0.7,    label="Individual runs"),
        # Line2D([0], [0], color="#555555",  linestyle="--",
        #        linewidth=0.9,              label="Reference (0)"),
    ]
    # ax.legend(handles=legend_elements, fontsize=8, frameon=False,
    #           loc="upper left", handlelength=1.6)

    # ── Title ────────────────────────────────────────────────────────────────
    if title:
        ax.set_title(title, fontsize=11, pad=10, fontweight="normal")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig



def compare_synthetic_model_aics(model_name, num_runs=100, save_path=None, show_legend=True, no_ridge=False):
    MODEL_COLORS = ['#2166AC', '#F4A736', '#2CA02C']  # blue, orange, green
    MODEL_NAMES  = ['1-Layer Linear', '2-Layer Linear', '2-Layer Nonlinear']
    MODEL_TO_IDX = {'1-Layer Linear': 0, '2-Layer Linear': 1, '2-Layer Nonlinear': 2}

    inference_model, depth, nln_name, _, _ = MODEL_CLASS_INFO[model_name]

    aic_data = []
    idxs = np.arange(num_runs)

    no_ridge_label = ''
    sessions_label = ''
    if no_ridge:
        no_ridge_label = 'no_ridge/'
        sessions_label = '_40_sessions'
    for i in idxs:
        results_dict = pkl.load(open(f'synthetic_data_GD_analysis/model_validation_experiments/{no_ridge_label}{inference_model}_{nln_name}_{depth}/{depth}_{inference_model}_{nln_name}_model{sessions_label}_{i}_multi_model_validation.pkl', 'rb'))

        aics = results_dict['aics']
        aic_data.append(aics)
    
    aic_data = np.array(aic_data)   # shape: (num_runs, num_models)

    true_idx = MODEL_TO_IDX[model_name]
    other_idxs = [idx for idx in [0, 1, 2] if idx != true_idx]

    model_success_rate = find_model_success_rate(aic_data, true_idx)
    print(f'{model_name} Network success rate: {model_success_rate}')

    true_model_aics = aic_data[:, MODEL_TO_IDX[model_name]]

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 6), sharey=True)
    
    axes[0].set_ylabel(r'$\Delta$AIC', fontsize=24)
    # axes[0].set_yticks([0, 1000, 2000, 3000])
    # axes[1].set_yticks([0, 1000, 2000, 3000, 4000])
    # axes[2].set_yticks([0, 1000, 2000, 3000, 4000])
    # plt.ylim(bottom=-100, top=3500)

    for idx, ax in zip(other_idxs, axes):
        jitter = rng.uniform(-0.25, 0.25, size=len(aic_data))

        winners = find_winners(aic_data, idx, true_idx)
        winners = np.asarray(winners)
        hits = (winners == true_idx)
        hit_rate = hits.sum() / len(aic_data)
        print(f'Idx {idx}: {hit_rate}')

        colors = [MODEL_COLORS[w] for w in winners]

        diffs = aic_data[:, idx] - true_model_aics
        ax.scatter(jitter, diffs, c=colors, s=60,
                   edgecolors='white', linewidths=0.3, zorder=3)

        ax.axhline(0, color='lightgray', linestyle='--', linewidth=0.9, zorder=2)
        # ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        # ax.set_ylim(bottom=-20)
        ax.locator_params(axis='y', nbins=4)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.tick_params(labelsize=23, length=3, width=0.8)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
        ax.set_axisbelow(True)


    if show_legend:
        legend_handles = [mpatches.Patch(color=c, label=n)
                      for c, n in zip(MODEL_COLORS, MODEL_NAMES)]
        
        fig.legend(
        handles=legend_handles,
        fontsize=17, frameon=False,
        loc='right',
        bbox_to_anchor=(1.01, 0.5),  # was 0.93, now sits just inside top of figure
        ncol=1,
        )

        plt.tight_layout(rect=[0, 0, 0.57, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


    

if __name__ == '__main__':
    # Experiments on 1-layer linear networks with no regularization
    # compare_synthetic_model_aics('1-Layer Linear', num_runs=100, save_path=None, show_legend=False, no_ridge=True)
    # run_model_aic_analysis('gd', 'linear', 'one-layer', num_runs=100, save_path=None, no_ridge=True)


    # Experiments on all models when using regularization
    for model_name in list(MODEL_CLASS_INFO.keys()):
        inference_model, depth, nln_name, _, _ = MODEL_CLASS_INFO[model_name]
        model_id = f'{inference_model}_{nln_name}_{depth}'
        save_path = f'synthetic_data_GD_analysis/model_validation_experiments/{model_id}_aic_jitterplot.pdf'

        compare_synthetic_model_aics(model_name, num_runs=100, save_path=None, show_legend=False, no_ridge=False)
        
        save_path = f'synthetic_data_GD_analysis/model_validation_experiments/{model_id}_nolegend.pdf'
        run_model_aic_analysis(inference_model, nln_name, depth, num_runs=100, save_path=None, no_ridge=False)

    print(1./0)

    # Recovery analysis
    recovery_types = ['alpha', 'beta', 'w0', 'full']
    for recovery_type in recovery_types:
        save_path = f'synthetic_data_GD_analysis/single_model/{recovery_type}_recovery.pdf'
        visualize_recovery_all_models(recovery_type=recovery_type, save_path=None) 

    for model_name in list(MODEL_CLASS_INFO.keys()):
        inference_model, depth, nln_name, _, _ = MODEL_CLASS_INFO[model_name]
        model_id = f'{inference_model}_{nln_name}_{depth}'

        for model_idx in range(100):
            weights_path = f'synthetic_data_GD_analysis/model_validation_experiments/inferred_weights/{model_id}/{model_idx}_inferred_vs_true_weights.pdf'
            visualize_inferred_weights_and_true_weights(model_name, model_idx, save_path=None, stride=1)


    

    # visualize_alpha_beta_recovery(inference_model, nln_name, depth)
    

    # results_dict = run_single_model_inference(inference_model, nln_name, depth, num_sessions=40, key=jr.PRNGKey(97), num_seeds=6, w0_init_scale=1e0)

    # print(results_dict['alpha_beta_error'])
    # print(results_dict['log_alpha_beta_error'])
    # model_idx = 54 # 1, 0

    # save_path = f'synthetic_data_GD_analysis/model_validation_experiments/{model_id}_{model_idx}_inferred_vs_true_weights.pdf'

    # visualize_inferred_weights_and_true_weights(inference_model, nln_name, depth, model_idx, save_path=save_path)

    # run_model_aic_analysis(inference_model, nln_name, depth, num_runs=100, plot_bar_graph=False, make_violin_plot=False, 
    #                        save_path=f'synthetic_data_GD_analysis/model_validation_experiments/{model_id}_nolegend.pdf')

    # true_model_name = 'one-layer linear gd'
    # num_sessions = 40
    # ntrials = 2
    # key = jr.PRNGKey(81)

    # # run_multi_model_inference(true_model_name, num_sessions, key, num_seeds=10, w0_init_scale=5e-1)
    # results = run_multi_model_inference(true_model_name, num_sessions, key, num_seeds=2, layer_two_multiplier=1., w0_init_scale=1e0)

    # print(results['rel_aics'])