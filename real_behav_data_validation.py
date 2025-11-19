import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax import debug
from jax.tree_util import tree_map
import numpy as np
import pickle as pkl

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd

from shallow_linear_net import shallow_net, shallow_net_params, fit_shallow_model
from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from utils import generate_multisession_graded_data, calculate_model_agnostic_bias, calculate_session_accs, flatten_params, turn_pytree_to_arr, realign_weight_trajectories, combine_batch_weights, compute_rightward_statistics, compute_two_layer_flattened_net_params
from simple_rates_and_weights_inference import *
from aic_single_trial_validation import *
from metric_GD_inference import run_randomized_metric_GD_inference_procedure, run_metric_GD_forward_trajectory


def calculate_mouse_session_accs(mouse_behav_dict):
    behav_df = mouse_behav_dict['behavior_df']
    session_accuracies = behav_df.groupby("session")["reward"].mean().to_numpy()
    return session_accuracies


def run_inferred_forward_trajectory(hyperparams_arr, inputs, outputs, rewards, model_type, learning_rule):
    assert learning_rule in ['gd', 'te', 'metric_gd'] 
    ws = None
    if learning_rule == 'metric_gd':
        _, ws = run_metric_GD_forward_trajectory(hyperparams_arr, inputs, outputs, rewards, model_type, batched=False, hyperparams_is_list=True)
    else:
        if model_type == 'two-layer':
            _, ws = run_deep_forward_weight_trajectory(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=learning_rule)
        else:
            _, ws = run_onelayer_forward_weight_trajectory(hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=learning_rule)
        
    return ws


def single_trial_metric_GD_data_validation(mouse_behav_dict, key, num_opt_seeds=20):
    _, key, skey = jr.split(key, 3)
    skeys = jr.split(skey, 6)

    print(f'FITTING TO REAL BEHAVIORAL DATA WITH METRIC GD NETWORK')
    model_types = ['two-layer', 'one-layer']
    test_learning_rule = 'metric_gd'

    nlls = np.zeros(2)
    ps = np.array([13, 9])
    inferred_hyperparams_collection = np.zeros((2, 13))

    inputs = mouse_behav_dict['inputs']
    outputs = mouse_behav_dict['outputs'].astype(jnp.int32)
    rewards = mouse_behav_dict['rewards'] 

    curr_key = skeys[5]
    for j in range(len(model_types)):
        model_type = model_types[j]
        _, curr_key = jr.split(curr_key, 2)

        print(f'\nPerforming inference in a {model_type} network with the {test_learning_rule} learning rule...')
        _, min_nll, hyperparams_arr = run_randomized_metric_GD_inference_procedure(inputs, outputs, rewards, curr_key, model_type=model_type,
                                                                                   num_seeds=num_opt_seeds, max_iters=500, tol=1e-4, batched=False,
                                                                                   alpha_init_scale=0.01, beta_init_scale=1., w0_init_scale=0.1)
        
        nlls[j] = min_nll
        inferred_hyperparams_collection[j, :len(hyperparams_arr)] = hyperparams_arr
        print(f'Inferred (training) NLL with {test_learning_rule} as the true learning rule: {min_nll}')

    AICs = 2 * nlls + 2 * ps

    print(f'\nNLLs: {nlls}')
    print(f'AICs: {AICs}')
    print(f'Inferred hyperparameters: {inferred_hyperparams_collection}')

    return nlls, AICs, inferred_hyperparams_collection


def single_trial_behav_data_validation(mouse_behav_dict, key, num_opt_seeds=15):
    _, key, skey = jr.split(key, 3)
    skeys = jr.split(skey, 6)

    print(f'FITTING TO REAL BEHAVIORAL DATA')
    model_types = ['two-layer', 'two-layer', 'one-layer', 'one-layer']
    test_learning_rules = ['gd', 'te', 'gd', 'te']

    nlls = np.zeros(4)
    ps = np.array([11, 11, 8, 8])
    inferred_hyperparams_collection = np.zeros((4, 11))

    inputs = mouse_behav_dict['inputs']
    outputs = mouse_behav_dict['outputs'].astype(jnp.int32)
    rewards = mouse_behav_dict['rewards'] 

    curr_key = skeys[5]
    for j in range(len(model_types)):
        model_type = model_types[j]
        test_learning_rule = test_learning_rules[j]
        _, curr_key = jr.split(curr_key, 2)

        print(f'\nPerforming inference in a {model_type} network with the {test_learning_rule} learning rule...')
        num_alpha_blocks = 1
        _, min_nll, hyperparams_arr = run_randomized_optimization_procedure(inputs, outputs, rewards, curr_key, learning_rule=test_learning_rule, model_type=model_type,
                                                                            num_seeds=num_opt_seeds, max_iters=1_000, 
                                                                            tol=1e-6, method='optax',
                                                                            alpha_init_scale=0.1, beta_init_scale=1., w0_init_scale=0.1, 
                                                                            num_alpha_blocks=num_alpha_blocks, l2_lambda=0., batched=False)
        nlls[j] = min_nll
        inferred_hyperparams_collection[j, :len(hyperparams_arr)] = hyperparams_arr
        print(f'Inferred (training) NLL with {test_learning_rule} as the true learning rule: {min_nll}')

    AICs = 2 * nlls + 2 * ps

    print(f'\nNLLs: {nlls}')
    print(f'AICs: {AICs}')
    print(f'Inferred hyperparameters: {inferred_hyperparams_collection}')

    return nlls, AICs, inferred_hyperparams_collection


def visualize_metric_GD_inferred_weight_trajectories(mouse_fit_results_dict, plot_both_left_right=True):
    model_types = ['two-layer', 'one-layer']
    learning_rule = 'metric_gd'
    true_num_params = [13, 9]
    n_alphas = [3, 2]

    fip = mouse_fit_results_dict['fip']
    inputs = mouse_fit_results_dict['inputs']
    outputs = mouse_fit_results_dict['outputs'].astype(jnp.int32)
    rewards = mouse_fit_results_dict['rewards']
    ma_bias = mouse_fit_results_dict['ma_bias']
    session_accs = mouse_fit_results_dict['session_accs']

    nlls = mouse_fit_results_dict['nlls']
    AICs = mouse_fit_results_dict['AICs']

    print(f'Results for mouse {fip}:')
    print(f'Model-agnostic bias: {ma_bias}')
    print(f'Session accuracies: {session_accs}')

    inferred_hyperparams_collection = mouse_fit_results_dict['all_inferred_hyperparams']

    for i in range(2):
        model_type = model_types[i]
        num_params = true_num_params[i]
        hyperparams_arr = inferred_hyperparams_collection[i][:num_params]
        nll = nlls[i]
        aic = AICs[i]

        print(f'\n{model_type} metric_gd alpha, beta: {jnp.exp(hyperparams_arr[:n_alphas[i]])}, {jnp.exp(hyperparams_arr[n_alphas[i]])}')
        print(f'Initial weights: {hyperparams_arr[2:]}')
        print(f'NLL: {nll}, AIC: {aic}')
        
        ws = run_inferred_forward_trajectory(hyperparams_arr, inputs, outputs, rewards, model_type, 'metric_gd')

        fig, axs = plt.subplots(1 + plot_both_left_right, 3, figsize=(18, 10), sharex=True, sharey=True)

        inferred_trajectories = flatten_params(ws)
        if model_type == 'two-layer':
            inferred_trajectories = compute_two_layer_flattened_net_params(inferred_trajectories)

        j = 0
        for ax in axs.ravel():
            if plot_both_left_right:
                ax.plot(inferred_trajectories[:, j], lw=3., color='black', linestyle='dashed')
            else:
                ax.plot(inferred_trajectories[:, j] - inferred_trajectories[:, j+3], lw=3., color='black', linestyle='dashed')
            ax.axhline(0, lw=2., color='blue', linestyle='dashed')
            j += 1
        plt.show()


def visualize_inferred_weight_trajectories(mouse_fit_results_dict, plot_both_left_right=True):
    model_types = ['two-layer', 'two-layer', 'one-layer', 'one-layer']
    learning_rules = ['gd', 'te', 'gd', 'te']
    true_num_params = [11, 11, 8, 8]

    fip = mouse_fit_results_dict['fip']
    inputs = mouse_fit_results_dict['inputs']
    outputs = mouse_fit_results_dict['outputs'].astype(jnp.int32)
    rewards = mouse_fit_results_dict['rewards']
    ma_bias = mouse_fit_results_dict['ma_bias']
    session_accs = mouse_fit_results_dict['session_accs']

    nlls = mouse_fit_results_dict['nlls']
    AICs = mouse_fit_results_dict['AICs']

    print(f'Results for mouse {fip}:')
    print(f'Model-agnostic bias: {ma_bias}')
    print(f'Session accuracies: {session_accs}')

    inferred_hyperparams_collection = mouse_fit_results_dict['all_inferred_hyperparams']

    for i in range(4):
        model_type = model_types[i]
        learning_rule = learning_rules[i]
        num_params = true_num_params[i]
        hyperparams_arr = inferred_hyperparams_collection[i][:num_params]
        nll = nlls[i]
        aic = AICs[i]

        print(f'\n{model_type} {learning_rule} alpha, beta: {jnp.exp(hyperparams_arr[0])}, {jnp.exp(hyperparams_arr[1])}')
        print(f'Initial weights: {hyperparams_arr[2:]}')
        print(f'NLL: {nll}, AIC: {aic}')
        
        ws = run_inferred_forward_trajectory(hyperparams_arr, inputs, outputs, rewards, model_type, learning_rule)

        fig, axs = plt.subplots(1 + plot_both_left_right, 3, figsize=(18, 10), sharex=True, sharey=True)

        inferred_trajectories = flatten_params(ws)
        if model_type == 'two-layer':
            inferred_trajectories = compute_two_layer_flattened_net_params(inferred_trajectories)

        j = 0
        for ax in axs.ravel():
            if plot_both_left_right:
                ax.plot(inferred_trajectories[:, j], lw=3., color='black', linestyle='dashed')
            else:
                ax.plot(inferred_trajectories[:, j] - inferred_trajectories[:, j+3], lw=3., color='black', linestyle='dashed')
            ax.axhline(0, lw=2., color='blue', linestyle='dashed')
            j += 1

        plt.show()


# def plot_inferred_bias_distribution(hyperparams_arr, inputs, num_trials, key):
#     alpha_star = jnp.exp(hyperparams_arr[0])
#     beta_star = jnp.exp(hyperparams_arr[1])
#     w0_inferred = hyperparams_arr[2:]
#     datasets, train_test_dict = generate_train_test_split(alpha_star, beta_star, true_learning_rule, true_model_type, session_info, key, 
#                                                           num_train_sets=0, num_test_sets=num_trials, vary_inputs=False, true_w0=w0_inferred, fixed_inputs=inputs)

#     print(train_test_dict)

#     new_w0_test_biases = train_test_dict['test_ma_biases']


def plot_single_trial_relative_aics(aic_data, fips, trial_idxs, 
                                    method_labels = {1: '2-layer TE', 2: '1-layer GD', 3: '1-layer TE'},
                                    ylims=None):
    A = aic_data    # num_trials x num_models
    num_test_items = A.shape[-1]
    num_trials = A.shape[0]

    # Prepare data for seaborn
    records = []
    for trial in trial_idxs:
        baseline_val = A[trial, 0]  # Baseline is the first method (e.g. 2-layer metric GD, or 2-layer GD)
        curr_fip = fips[trial]
        for method in range(1, num_test_items):
            records.append({
                'Mouse': f'Mouse {curr_fip}',
                'Method': method_labels.get(method, f'Method {method}'),
                'Relative AIC': A[trial, method] - baseline_val,
                'Baseline': baseline_val - baseline_val  # This will always be 0
            })

    df = pd.DataFrame(records)

    # Plot: use stripplot instead of boxplot
    plt.figure(figsize=(13, 5))
    sns.stripplot(data=df, x='Mouse', y='Relative AIC', hue='Method', dodge=True, size=14, palette='tab10', jitter=False)

    # Overlay black bars for ground truth
    # For each test set, draw a horizontal line at the baseline
    xticks = plt.gca().get_xticks()
    trials = df['Mouse'].unique()
    for i, trial in enumerate(trials):
        y = df[df['Mouse'] == trial]['Baseline'].iloc[0]
        plt.hlines(y, i - 0.33, i + 0.33, color='black', linewidth=2.5)

    plt.title(f"Single Trial Goodness of Fit", fontsize=20)
    plt.ylabel('Relative AIC')
    plt.xlabel("Mouse")
    plt.legend(title='Method')
    plt.tight_layout()

    if ylims is not None:
        plt.ylim(ylims)

    plt.show()


if __name__ == "__main__":

    expert_threshold = 0.7
    fips = np.load('all_mice_IBL_fips.npy')
    n = len(fips)

    model_types = ['two-layer', 'one-layer']
    learning_rule = 'metric_gd'

    # relative_aic_diffs = np.zeros((n, 4))
    expert_reached_idxs = []

    # gd_alphas = []
    # gd_betas = []

    # te_alphas = []
    # te_betas = []

    mgd_alpha_W1s = []
    mgd_alpha_W2_consts = []
    mgd_alpha_W2_stims = []
    mgd_betas = []

    ma_biases = []

    plot_idx = 0

    all_nlls = np.zeros((n, 6))
    all_aics = np.zeros((n, 6))

    for fip_idx in range(n):
        fip = fips[fip_idx]

        metric_GD_mouse_fit_results_dict = pkl.load(open(f'metric_GD_mouse_IBL_single_trial_fits/fip{fip}_results.pkl', 'rb'))
        session_accs = metric_GD_mouse_fit_results_dict['session_accs']
        expert = (session_accs.max() >= expert_threshold)

        gd_te_fit_results_dict = pkl.load(open(f'mouse_IBL_single_trial_fits/fip{fip}_results.pkl', 'rb'))

        metric_GD_aics = metric_GD_mouse_fit_results_dict['AICs']
        gd_te_aics = gd_te_fit_results_dict['AICs']
        print(metric_GD_aics)
        print(gd_te_aics)
        print('\n')

        all_nlls[fip_idx, :2] = metric_GD_mouse_fit_results_dict['nlls']
        all_nlls[fip_idx, 2:] = gd_te_fit_results_dict['nlls']

        all_aics[fip_idx, :2] = metric_GD_aics
        all_aics[fip_idx, 2:] = gd_te_aics

        if not expert:
            continue
        else:
            expert_reached_idx = np.where(session_accs >= expert_threshold)[0][0]
            expert_reached_idxs.append(expert_reached_idx)

            ma_bias = metric_GD_mouse_fit_results_dict['ma_bias']
            ma_biases.append(ma_bias)

            all_hyperparams = metric_GD_mouse_fit_results_dict['all_inferred_hyperparams']
            two_layer_hyperparams = all_hyperparams[0]
            # one_layer_hyperparams = all_hyperparams[1, :10]

            mgd_alpha_W1s.append(np.exp(two_layer_hyperparams[0]))
            mgd_alpha_W2_consts.append(np.exp(two_layer_hyperparams[1]))
            mgd_alpha_W2_stims.append(np.exp(two_layer_hyperparams[2]))
            mgd_betas.append(np.exp(two_layer_hyperparams[3]))

            # if plot_idx < 5:
            #     visualize_metric_GD_inferred_weight_trajectories(metric_GD_mouse_fit_results_dict, plot_both_left_right=False)
            #     plot_idx += 1

    start = 10
    num_trials = 15

    min_idxs = np.argmin(all_aics, axis=1)
    num_best_metric_GD = np.sum(min_idxs < 1)
    print(num_best_metric_GD)

    num_best_2layer_GD = np.sum((min_idxs >= 2) & (min_idxs < 3))
    num_best_2layer_TE = np.sum((min_idxs >= 3) & (min_idxs < 4))
    print(num_best_2layer_GD)
    print(num_best_2layer_TE)

    plot_single_trial_relative_aics(all_aics, fips, trial_idxs=np.arange(start, start+num_trials), 
                                    method_labels = {1: '1-layer m-GD', 2: '2-layer GD', 3: '2-layer TE', 4: '1-layer GD', 5: '1-layer TE'},
                                    ylims=(-50, 200))
    
    metric_gd_minus_true_gd = all_aics[:, 0] - all_aics[:, 2]
    metric_gd_minus_true_te = all_aics[:, 0] - all_aics[:, 3]

    plt.figure()
    plt.hist(metric_gd_minus_true_gd, bins=20, alpha=0.5, label='m-GD - true GD')
    plt.hist(metric_gd_minus_true_te, bins=20, alpha=0.5, label='m-GD - true TE')
    plt.axvline(0, color='black', linestyle='dashed', lw=2.)
    plt.xlabel('AIC Difference')
    plt.ylabel('Number of Mice')
    plt.legend()
    plt.title('AIC Differences between Metric GD and True Learning Rules')
    plt.show()


    expert_reached_idxs = np.asarray(expert_reached_idxs)
    k = 5
    top_indices = np.argpartition(expert_reached_idxs, 5)[:5]
    top_indices = top_indices[np.argsort(expert_reached_idxs[top_indices])]

    bottom_indices = np.argpartition(expert_reached_idxs, -k)[-k:]

    # gd_alphas = np.asarray(gd_alphas)
    # gd_betas = np.asarray(gd_betas)
    # te_alphas = np.asarray(te_alphas)
    # te_betas = np.asarray(te_betas) 

    ma_biases = np.asarray(ma_biases)

    top_expert_idxs = expert_reached_idxs[top_indices]
    bottom_expert_idxs = expert_reached_idxs[bottom_indices]

    print(top_expert_idxs)
    print(bottom_expert_idxs)
    print('\n')

    mgd_alpha_W1s = np.asarray(mgd_alpha_W1s)
    mgd_alpha_W2_consts = np.asarray(mgd_alpha_W2_consts)
    mgd_alpha_W2_stims = np.asarray(mgd_alpha_W2_stims)
    mgd_betas = np.asarray(mgd_betas)

    top_mgd_alpha_W1s = mgd_alpha_W1s[top_indices]
    top_mgd_alpha_W2_consts = mgd_alpha_W2_consts[top_indices]
    top_mgd_alpha_W2_stims = mgd_alpha_W2_stims[top_indices]
    top_mgd_betas = mgd_betas[top_indices]

    bottom_mgd_alpha_W1s = mgd_alpha_W1s[bottom_indices]
    bottom_mgd_alpha_W2_consts = mgd_alpha_W2_consts[bottom_indices]
    bottom_mgd_alpha_W2_stims = mgd_alpha_W2_stims[bottom_indices]
    bottom_mgd_betas = mgd_betas[bottom_indices]

    top_ma_biases = ma_biases[top_indices]
    bottom_ma_biases = ma_biases[bottom_indices]
    print(top_ma_biases)
    print(bottom_ma_biases)
            
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    color = 'green'
    ax[0, 0].scatter(expert_reached_idxs, mgd_alpha_W1s, color=color, s=30)
    ax[0, 0].scatter(top_expert_idxs, top_mgd_alpha_W1s, c=color, edgecolors="black", s=120, marker="*", label="Special points")
    ax[0, 0].scatter(bottom_expert_idxs, bottom_mgd_alpha_W1s, c=color, edgecolors="darkviolet", s=120, marker="*", label="Special points")
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_xlabel('# Sessions to Reach Expert Status')
    ax[0, 0].set_ylabel('alpha W1')

    ax[0, 1].scatter(expert_reached_idxs, mgd_alpha_W2_consts, color=color, s=30)
    ax[0, 1].scatter(top_expert_idxs, top_mgd_alpha_W2_consts, c=color, edgecolors="black", s=120, marker="*", label="Special points")
    ax[0, 1].scatter(bottom_expert_idxs, bottom_mgd_alpha_W2_consts, c=color, edgecolors="darkviolet", s=120, marker="*", label="Special points")
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_xlabel('# Sessions to Reach Expert Status')
    ax[0, 1].set_ylabel('alpha W2 const')

    ax[1, 0].scatter(expert_reached_idxs, mgd_alpha_W2_stims, color=color, s=30)
    ax[1, 0].scatter(top_expert_idxs, top_mgd_alpha_W2_stims, c=color, edgecolors="black", s=120, marker="*", label="Special points")
    ax[1, 0].scatter(bottom_expert_idxs, bottom_mgd_alpha_W2_stims, c=color, edgecolors="darkviolet", s=120, marker="*", label="Special points")
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_xlabel('# Sessions to Reach Expert Status')
    ax[1, 0].set_ylabel('alpha W2 stim')

    ax[1, 1].scatter(expert_reached_idxs, mgd_betas, color=color, s=30)
    ax[1, 1].scatter(top_expert_idxs, top_mgd_betas, c=color, edgecolors="black", s=120, marker="*", label="Special points")
    ax[1, 1].scatter(bottom_expert_idxs, bottom_mgd_betas, c=color, edgecolors="darkviolet", s=120, marker="*", label="Special points")
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlabel('# Sessions to Reach Expert Status')
    ax[1, 1].set_ylabel('beta')

    # ax[2].scatter(te_alphas, te_betas, color=color, s=30)
    # ax[2].scatter(top_te_alphas, top_te_betas, c=color, edgecolors="black", s=120, marker="*", label="Special points")
    # ax[2].scatter(bottom_te_alphas, bottom_te_betas, c=color, edgecolors="darkviolet", s=120, marker="*", label="Special points")
    # ax[2].set_xscale('log')
    # ax[2].set_yscale('log')
    # ax[2].set_xlabel('alpha')
    # ax[2].set_ylabel('beta')

    plt.show()


    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.scatter(ma_biases, expert_reached_idxs, color='blue', s=30)
    ax.scatter(top_ma_biases, top_expert_idxs, c="blue", edgecolors="black", s=120, marker="*", label="Special points")
    ax.scatter(bottom_ma_biases, bottom_expert_idxs, c="blue", edgecolors="purple", s=120, marker="*", label="Special points")
    ax.set_ylabel('# Sessions to Reach Expert Status')
    ax.set_xlabel('Model-agnostic bias')

    plt.show()


    #     relative_aic_diffs[fip_idx] = AICs - AICs[0]

    # plot_single_trial_relative_aics(relative_aic_diffs, fips, trial_idxs=np.arange(15, 20))



    # key = jr.PRNGKey(314)
    # _, key = jr.split(key, 2)

    # all_mice_behav_dict = pkl.load(open('all_mice_IBL_behavioral_data.pkl', 'rb'))

    # for i in range(len(fips)):
    #     # testing single-trial fits for a right-biased mouse
    #     curr_fip = fips[i]  
    #     # print(fips)
    #     mouse_behav_dict = all_mice_behav_dict[curr_fip]
    #     # print(mouse_behav_dict)

    #     inputs = mouse_behav_dict['inputs']
    #     outputs = mouse_behav_dict['outputs']
    #     rewards = mouse_behav_dict['rewards']

    #     rightward_stats = compute_rightward_statistics(inputs, outputs)
    #     ma_bias = calculate_model_agnostic_bias(rightward_stats, zero_contrast_fraction=0.)
    #     print(ma_bias)

    #     curr_session_accs = calculate_mouse_session_accs(mouse_behav_dict)
    #     print(curr_session_accs)

    #     nlls, AICs, inferred_hyperparams_collection = single_trial_behav_data_validation(mouse_behav_dict, key, num_opt_seeds=40)
