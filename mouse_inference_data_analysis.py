
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
import numpy as np
import pickle as pkl
import scipy

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from matplotlib.ticker import StrMethodFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import seaborn as sns
import pandas as pd
import os 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from utils import flatten_params, compute_two_layer_flattened_net_params, compute_rightward_statistics, calculate_model_agnostic_bias, plot_psychometrics, visualize_weight_trajectories
from utils import generate_per_session_psychometrics, calculate_normalized_avg_acc, calculate_trial_averaged_acc
from utils import run_linear_regression, get_biases_per_session, calculate_simple_mouse_ma_bias, get_biases_per_session_batched, get_nonlinearity
from utils import visualize_weight_trajectories, calculate_accs_per_session, calculate_batched_quantity_per_session, calculate_running_avg_quantity
from utils import compute_coeff_of_determination, get_zero_contrast_mask, compute_variances_along_trajectory, find_winners, get_sigmoid_fit
from utils import compute_r2_distribution, visualize_r2_distributions, get_pearson_correlations, vectorized_generate_per_session_psychometrics
from inference_utils import visualize_channel_contributions, visualize_nln_usage, visualize_alpha_beta_nlls
from inference_utils import extract_weights_from_hyperparams, get_num_alphas, get_depth, model_forward, batched_model_forward, convert_weights_to_dict
from inference_utils import train_gd_network, gather_input_contributions
from inference_pipeline import run_single_learning_forward_trajectory, compute_nll, compute_hessian, run_epsilon_ball_test, finite_diff_gradient
from globals import (DATASET_TO_CHOICE_KEYWORD, DEPTH_TO_NUM_WEIGHTS, EXPERT_THRESHOLD, MODEL_CLASS_INFO,
                     MODEL_PLOT_NAMES, MU_ALPHA, MU_BETA, NUM_FINAL_SESSIONS, NUM_GRADING_LEVELS,
                     NUM_ONE_LAYER_WEIGHTS, NUM_TWO_LAYER_WEIGHTS, RIDGE_ALPHABETA, RIDGE_LAMBDA,
                     SAXE_BAD_IDXS, WITTEN_DATA_CONTRAST_LEVELS)
from metric_GD_network import fit_metric_GD_net, generate_metric_GD_train_test_split


def load_data(dataset_name, remove_bad_idxs=True):
    assert dataset_name in ['witten', 'saxe'], f'Invalid dataset name: {dataset_name}'
    with open(f'{dataset_name}_data_GD_analysis_final/{dataset_name}_mouse_data.pkl', 'rb') as f:
        data = pkl.load(f)

    if dataset_name == 'saxe' and remove_bad_idxs:
        data = {k: v for k, v in data.items() if k not in SAXE_BAD_IDXS}

    return data


def get_mouse_session_accs(mouse_dict):
    mask = get_zero_contrast_mask(mouse_dict['inputs'])
    rewards = mouse_dict['rewards']
        
    trials_per_session = mouse_dict['trials_per_session']
    session_accs = calculate_batched_quantity_per_session(rewards, trials_per_session, mask=mask)
    return session_accs


def find_best_hyperparams(optim_dict):
    convergence_failed=False
    all_nlls = jnp.asarray(optim_dict['nlls'])
    bad_run_idxs = jnp.asarray(optim_dict['bad_run_idxs']).astype(int)

    if len(bad_run_idxs) > 0:
        bad_nlls = all_nlls[bad_run_idxs]
        if jnp.min(bad_nlls) == jnp.min(all_nlls):
            print('NOTE: HERE WE ARE USING A NON-CONVERGED HYPERPARAM VECTOR!', flush=True)
            convergence_failed=True
        
    all_hyperparams = optim_dict['hyperparams']
    best_idx = jnp.argmin(all_nlls)
    
    return all_hyperparams[best_idx], convergence_failed


def get_sampling_tag(sampling_type):
    assert sampling_type in ['deterministic', 'MAP']
    if sampling_type == 'deterministic':
        return ''
    return '_MAP_Laplace'


def get_mouse_aics(dataset_name, mouse_id):
    mouse_aics = []
    mouse_nlls = []
    for key, value in MODEL_CLASS_INFO.items():
        inference_model, depth, nln_name, num_params, _ = value 
        nln = get_nonlinearity(nln_name)
    
        results_dict = pkl.load(open(f'{dataset_name}_data_GD_analysis_final/fitted_models/{inference_model}_{nln_name}_{depth}/mouse_{mouse_id}_results_{depth}_{inference_model}_{nln_name}.pkl', 'rb'))
        x0, _ = find_best_hyperparams(results_dict['optim_dict'])
        inputs = results_dict['inputs']
        outputs = results_dict['outputs'].astype(jnp.int32)
        rewards = results_dict['rewards']
        nll = compute_nll(x0, inputs, outputs, rewards, nln, inference_model, batched=False, alpha_W1=None, num_weights=DEPTH_TO_NUM_WEIGHTS[depth], ridge_lambda=0., alphabeta_ridge=0.)
        
        mouse_aics.append(2 * nll + 2 * num_params)
        mouse_nlls.append(nll)
    
    return mouse_aics, mouse_nlls

def plot_linear_correlation(arr1, arr2, xlabel, ylabel, filepath=None, figsize=(5, 4), labelsize=14, ticksize=16):
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(arr1, arr2, color="#081523", s=100, alpha=1.0,
            edgecolors='white', linewidths=0.3, zorder=3)

    x_line, y_line, r_squared = run_linear_regression(arr1, arr2)
    ax.plot(x_line, y_line, color='#D6604D', lw=2.2, linestyle='--', zorder=2)

    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    print(f'R^2 value for {xlabel} vs {ylabel}: {r_squared}')

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.tick_params(labelsize=ticksize, length=3, width=0.8)
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=3)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if filepath is None:
        plt.show()
    else:
        fig.savefig(f'{filepath}/linear_plot_{xlabel}_vs_{ylabel}.pdf',
                    dpi=300, bbox_inches='tight')


def plot_top_PCs(top_eigvecs, filepath=None):
    COLORS = ['#00838F', '#AD1457', '#F9A825']  # teal, magenta, gold
    LABELS = ['PC 1', 'PC 2', 'PC 3']
    OFFSET = 2.5  # vertical separation between traces; tune to taste

    fig, ax = plt.subplots(figsize=(6, 5))
    xs = np.arange(1, len(top_eigvecs[0]) + 1)

    for i, (eigvec, color, label) in enumerate(zip(top_eigvecs[:3], COLORS, LABELS)):
        offset = (2-i) * OFFSET 
        baseline_y = offset

        # Stem lines
        for x, y in zip(xs, eigvec):
            ax.plot([x, x], [baseline_y, baseline_y + y], color=color, lw=1.2, alpha=0.9)

        # Markers
        ax.scatter(xs, baseline_y + eigvec, color=color, s=70, zorder=3)

        # Baseline
        ax.axhline(baseline_y, color=color, linewidth=0.6, linestyle='-', alpha=0.4, zorder=0)

        # Label on the left
        ax.text(-0.4, baseline_y, label, color=color, fontsize=22,
                ha='right', va='center', fontweight='bold')

    ax.set_xlabel('Feature Index', fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(labelsize=20, length=3, width=0.8)
    ax.xaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()


    if filepath is None:
        plt.show()
    else:
        fig.savefig(f'{filepath}/top_hyperparam_PCs.pdf', dpi=300, bbox_inches='tight')

        

def plot_PC_heatmap(plot_pc1, plot_pc2, plot_pc_idx1, plot_pc_idx2, color_feature, feature_name, filepath=None, cmap=None):
    assert feature_name in ['avg_acc_score', 'ma_bias']
    label = 'Normalized Accuracy'
    if feature_name == 'ma_bias':
        label = '|Bias|'
    
    if cmap is None:
        cmap = 'viridis'

    fig, ax = plt.subplots(figsize=(5, 4))

    scatter = ax.scatter(plot_pc1, plot_pc2,
                        c=color_feature, cmap=cmap,
                        s=100, edgecolors='white', linewidths=0.3, zorder=3)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=20)
    cbar.ax.tick_params(labelsize=19)

    ax.set_xlabel(f'PC {plot_pc_idx1}', fontsize=20)
    ax.set_ylabel(f'PC {plot_pc_idx2}', fontsize=20)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.tick_params(labelsize=19, length=3, width=0.8)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.xaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if filepath is None:
        plt.show()
    else:
        fig.savefig(f'{filepath}/{feature_name}_PC_heatmap.pdf', dpi=300, bbox_inches='tight')


def check_expert_status(mouse_dict, threshold=EXPERT_THRESHOLD, num_final_sessions=NUM_FINAL_SESSIONS):
    session_accs = get_mouse_session_accs(mouse_dict)
    final_accs = session_accs[-num_final_sessions:]
    avg_final_acc = final_accs.mean()
    return avg_final_acc >= threshold


def visualize_mouse_inferred_weights(dataset_name, mouse_id, save_path=None, stride=1):
    all_mice_dict = pkl.load(open(f'{dataset_name}_data_GD_analysis_final/{dataset_name}_mouse_data.pkl', 'rb'))
    mouse_dict = all_mice_dict[mouse_id]

    inputs = mouse_dict['inputs']
    outputs = mouse_dict[DATASET_TO_CHOICE_KEYWORD[dataset_name]].astype(jnp.int32)
    rewards = mouse_dict['rewards']
    mask = get_zero_contrast_mask(inputs)

    num_trials = len(inputs)
    trials = np.arange(1, num_trials+1)

    ma_bias = calculate_simple_mouse_ma_bias(outputs)
    print(f'Mouse {mouse_id}:')
    print(f'M-A bias for this model: {ma_bias}')

    running_accs = calculate_running_avg_quantity(rewards, window=200, mask=mask)
    print(f'Final running accuracies for this model: {running_accs[-10:]}')
    print('\n')

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True)
    fig.suptitle('Decision Contributions from Each Input Channel', fontsize=17, y=0.99)

    col_titles = ['constant channel', 'right channel', 'left channel']  
    for j, title in enumerate(col_titles):
        ax[j].set_title(title, fontsize=15)

    for i, key in enumerate(MODEL_CLASS_INFO):
        inference_model, depth, nln_name, num_params, color = MODEL_CLASS_INFO[key]
        results_dict = pkl.load(open(f'{dataset_name}_data_GD_analysis_final/fitted_models/{inference_model}_{nln_name}_{depth}/mouse_{mouse_id}_results_{depth}_{inference_model}_{nln_name}.pkl', 'rb'))
        hyperparams, _ = find_best_hyperparams(results_dict['optim_dict'])

        nln = get_nonlinearity(nln_name)
        _, ws, _ = run_single_learning_forward_trajectory(hyperparams, inputs, outputs, rewards,
                                                          nln, inference_model, alpha_W1=None)
        ws_flat = flatten_params(ws)
        loading_stats = gather_input_contributions(ws_flat, nln)
        right_loading_diffs = loading_stats['right_diffs']
        left_loading_diffs = loading_stats['left_diffs']

        ax[0].plot(trials[::stride], right_loading_diffs[:, 0][::stride], color=color, lw=1.8, linestyle='solid', label=key)    # constant channel Q-val diffs
        ax[1].plot(trials[::stride], right_loading_diffs[:, 1][::stride], color=color, lw=1.8, linestyle='solid', label=key)    # right channel Q-val diffs
        ax[2].plot(trials[::stride], left_loading_diffs[:, 2][::stride], color=color, lw=1.8, linestyle='solid')                # left channel Q-val diffs

    # Shared zero-line and spine cleanup across all axes
    for a in ax.flat:
        a.axhline(0, lw=0.9, color='gray', linestyle=':', zorder=0)
        a.spines[['top', 'right']].set_visible(False)
        a.spines[['left', 'bottom']].set_linewidth(0.8)
        a.locator_params(axis='y', nbins=4)

        xticks = np.arange(0, num_trials+1, 5000)
        a.set_xticks(xticks)
        a.set_xticklabels([f'{n:,}' for n in xticks], fontsize=12)

        a.tick_params(labelsize=12, length=3, width=0.8)
        a.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
        a.set_axisbelow(True)

    Q = r'$Q_R - Q_L$'

    ax[0].set_ylabel(f'{Q} \nContribution', fontsize=15)
    for j in range(3):
        ax[j].set_xlabel('Trial', fontsize=15)

    legend_elements = [
        Line2D([0], [0], color=MODEL_CLASS_INFO['1-Layer Linear'][4],  lw=1.8, linestyle='solid', label='1-Layer Linear'),
        Line2D([0], [0], color=MODEL_CLASS_INFO['2-Layer Linear'][4],  lw=1.8, linestyle='solid', label='2-Layer Linear'),
        Line2D([0], [0], color=MODEL_CLASS_INFO['2-Layer Nonlinear'][4],     lw=1.8, linestyle='solid', label='2-Layer Nonlinear'),
    ]
    fig.legend(
        handles=legend_elements,
        fontsize=14, frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(legend_elements),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

    return


def plot_nll_diff_histogram(nlls1, nlls2, filename):
    nll_diff = nlls1 - nlls2
    plt.figure(figsize=(8,6))
    plt.hist(nll_diff, bins=10, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.title('Histogram of NLL Differences (2-layer GD erf - GD linear)', fontsize=16)
    plt.xlabel('NLL Difference', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig(f'{filename}.png')


def extract_fitted_params(mouse_results_dict, inference_model, return_dict=False):
    hyperparams, _ = find_best_hyperparams(mouse_results_dict['optim_dict'])
    w0 = extract_weights_from_hyperparams(hyperparams, inference_model, return_dict=return_dict)
    depth = get_depth(hyperparams, inference_model)
    num_alphas = get_num_alphas(inference_model, depth)
    log_alphas = hyperparams[:num_alphas]
    lrs = jnp.exp(log_alphas)
    beta = jnp.exp(hyperparams[num_alphas])
    return w0, depth, lrs, beta


def extract_lrs_beta_and_w0(hyperparams, inference_model, return_dict=False):
    w0 = extract_weights_from_hyperparams(hyperparams, inference_model, return_dict=return_dict)
    depth = get_depth(hyperparams, inference_model)
    num_alphas = get_num_alphas(inference_model, depth)
    log_alphas = hyperparams[:num_alphas]
    lrs = jnp.exp(log_alphas)
    beta = jnp.exp(hyperparams[num_alphas])
    return lrs, beta, w0


def find_best_matching_model_outputs(true_outputs, model_outputs, top_k=5):
    # find the model output trajectory that minimizes the sum of absolute differences in outputs across all trials
    output_diffs = np.sum(np.abs(model_outputs - true_outputs.reshape(1, -1)), axis=1)  # shape: num_models
    best_model_idx = np.argmin(output_diffs)
    best_model_output = model_outputs[best_model_idx]

    # find k best model outputs
    best_k_model_idxs = np.argsort(output_diffs)[:top_k]
    k_best_outputs = model_outputs[best_k_model_idxs, :]

    return best_model_output, k_best_outputs, best_k_model_idxs


def get_running_accs_from_best_matching_outputs(true_outputs, all_model_outputs, targets, top_k=5, window=50):
    _, k_best_outputs, _ = find_best_matching_model_outputs(true_outputs, all_model_outputs, top_k=top_k)
    k_best_outputs = jnp.array(k_best_outputs).astype(jnp.int32)
    k_best_rewards = (k_best_outputs == targets.reshape(1, -1)).astype(jnp.int32)

    running_accs = calculate_running_avg_quantity(k_best_rewards, window)

    avg_running_accs = np.mean(running_accs, axis=0)
    min_running_accs = np.min(running_accs, axis=0)
    max_running_accs = np.max(running_accs, axis=0)
    std_running_accs = np.std(running_accs, axis=0)
    sem_running_accs = std_running_accs / np.sqrt(running_accs.shape[0])

    return avg_running_accs, min_running_accs, max_running_accs, std_running_accs, sem_running_accs


def get_model_session_accs_from_best_matching_outputs(true_outputs, all_model_outputs, targets, trials_per_session, by_bias=False, top_k=5):
    best_outputs, k_best_outputs, _ = find_best_matching_model_outputs(true_outputs, all_model_outputs, top_k=top_k)
    best_outputs = jnp.array(best_outputs).astype(jnp.int32)
    k_best_outputs = jnp.array(k_best_outputs).astype(jnp.int32)

    best_rewards = (best_outputs == targets).astype(jnp.int32)
    k_best_rewards = (k_best_outputs == targets.reshape(1, -1)).astype(jnp.int32)

    best_accs = calculate_accs_per_session(best_rewards, trials_per_session)
    k_best_accs = calculate_batched_quantity_per_session(k_best_rewards, trials_per_session)

    avg_k_accs = np.mean(k_best_accs, axis=0)
    # std_k_accs = np.std(k_best_accs, axis=0)
    min_k_accs = np.min(k_best_accs, axis=0)
    max_k_accs = np.max(k_best_accs, axis=0)

    return best_accs, avg_k_accs, min_k_accs, max_k_accs


def get_model_session_biases_from_best_matching_outputs(true_outputs, all_model_outputs, trials_per_session, top_k=5):
    # true_biases_within_session = get_biases_per_session(true_outputs, trials_per_session, calculate_within_session=True)
    best_outputs, k_best_outputs, _ = find_best_matching_model_outputs(true_outputs, all_model_outputs, trials_per_session, by_bias=False, top_k=top_k)
    best_outputs = jnp.array(best_outputs).astype(jnp.int32)
    k_best_outputs = jnp.array(k_best_outputs).astype(jnp.int32)

    best_model_biases_per_session = get_biases_per_session(best_outputs, trials_per_session, calculate_within_session=True)

    model_biases_per_session = np.zeros((top_k, len(trials_per_session)))

    for i in range(top_k):
        curr_outputs = k_best_outputs[i]
        model_biases_per_session[i] = get_biases_per_session(curr_outputs, trials_per_session, calculate_within_session=True)
    
    avg_model_biases_within_session = np.mean(model_biases_per_session, axis=0)
    min_model_biases_within_session = np.min(model_biases_per_session, axis=0)
    max_model_biases_within_session = np.max(model_biases_per_session, axis=0)

    return best_model_biases_per_session, avg_model_biases_within_session, min_model_biases_within_session, max_model_biases_within_session


def get_model_lls_from_best_matching_outputs(all_inferred_hyperparams, inputs, outputs, rewards, all_model_outputs, 
                                             nln, inference_model, trials_per_session, top_k=5):
    _, k_best_outputs, k_best_model_idxs = find_best_matching_model_outputs(outputs, all_model_outputs, trials_per_session, by_bias=False, top_k=top_k)

    k_best_outputs = jnp.array(k_best_outputs).astype(jnp.int32)
    k_best_hyperparams = jnp.array(all_inferred_hyperparams[k_best_model_idxs, :])

    k_best_lls = []

    for i in range(top_k):
        _, _, lls = run_single_learning_forward_trajectory(k_best_hyperparams[i], inputs, outputs, rewards, nln, inference_model, alpha_W1=None)
        k_best_lls.append(lls)
    
    k_best_lls = np.asarray(k_best_lls)

    k_best_session_lls = calculate_batched_quantity_per_session(k_best_lls, trials_per_session)

    avg_k_session_lls = np.mean(k_best_session_lls, axis=0)
    min_k_session_lls = np.min(k_best_session_lls, axis=0)
    max_k_session_lls = np.max(k_best_session_lls, axis=0)

    return avg_k_session_lls, min_k_session_lls, max_k_session_lls



# def compute_squared_RPEs(inferred_hyperparams, inputs, outputs, rewards, nln, inference_model, trials_per_session=None):
#     _, ws, _ = run_single_learning_forward_trajectory(inferred_hyperparams, inputs, outputs, rewards, 
#                                                    nln, inference_model, alpha_W1=None)
    
#     all_outs = batched_model_forward(ws, inputs, nln)   # shape: T x 2
#     preds = all_outs[jnp.arange(len(outputs)), outputs]  # shape: T
#     RPEs = (rewards - preds)**2
#     avg_RPEs_by_session = None 

#     if trials_per_session is not None:
#         session_endpts = np.cumsum(trials_per_session)
#         avg_RPEs_by_session = []
#         start_idx = 0
#         for end_idx in session_endpts:
#             session_RPEs = RPEs[start_idx:end_idx]
#             avg_session_RPE = jnp.mean(session_RPEs)
#             avg_RPEs_by_session.append(avg_session_RPE)
#             start_idx = end_idx
#         avg_RPEs_by_session = jnp.array(avg_RPEs_by_session)
    
#     return RPEs, avg_RPEs_by_session


def calculate_slope_fixed_grading(grading, inputs, outputs, trials_per_session, num_grading_levels, nonzero_gradings=WITTEN_DATA_CONTRAST_LEVELS):
    # psychometrics, _, _, _ = generate_per_session_psychometrics(inputs, outputs, trials_per_session, num_grading_levels=num_grading_levels, specific_grading_levels=nonzero_gradings)
    psychometrics, _, _, _ = vectorized_generate_per_session_psychometrics(inputs, outputs, trials_per_session, num_grading_levels=num_grading_levels, specific_grading_levels=nonzero_gradings)
    num_contrasts = psychometrics.shape[1]

    nonzero_gradings = np.asarray(nonzero_gradings)
    grading_idx = np.where(nonzero_gradings == grading)[0][0] 

    slope_diff = 0.
    if num_contrasts % 2 == 1:
        print(1./0)
        middle_idx = int(num_contrasts/2)
        slope_diff = (psychometrics[:, middle_idx + (grading_idx + 1)] - psychometrics[:, middle_idx - (grading_idx + 1)]) / (2 * grading)
    else:
        left_start = int(num_contrasts/2)-1
        right_start = int(num_contrasts/2)
        slope_diff = (psychometrics[:, right_start + grading_idx] - psychometrics[:, left_start - grading_idx]) / (2 * grading)

    return slope_diff


def calculate_model_slope_fixed_grading(grading, inputs, all_model_outputs, trials_per_session, num_grading_levels, nonzero_gradings=WITTEN_DATA_CONTRAST_LEVELS):
    all_slope_diffs = []
    for i in range(all_model_outputs.shape[0]):
        slope_diffs = calculate_slope_fixed_grading(grading, inputs, all_model_outputs[i], trials_per_session, num_grading_levels, nonzero_gradings)
        all_slope_diffs.append(slope_diffs)
    
    return jnp.asarray(all_slope_diffs)


def calculate_slope_differences_by_session(inputs, outputs, trials_per_session, min_contrast, max_contrast, num_grading_levels):
    psychometrics, _, _, _ = generate_per_session_psychometrics(inputs, outputs, trials_per_session, num_grading_levels=num_grading_levels)
    num_contrasts = psychometrics.shape[1]

    # middle_vals = None
    # if num_contrasts % 2 == 1:
    #     middle_vals = psychometrics[:, int(num_contrasts/2)]
    # else:
    #     middle_vals = 0.5 * (psychometrics[:, int(num_contrasts/2)-1] + psychometrics[:, int(num_contrasts/2)])
    
    right_slope = (psychometrics[:, -1] - psychometrics[:, -2]) / (max_contrast - min_contrast)
    left_slope = (psychometrics[:, 1] - psychometrics[:, 0]) / (max_contrast - min_contrast)
    slope_diffs_by_session = right_slope - left_slope

    return slope_diffs_by_session

def calculate_model_slope_diffs(inputs, all_model_outputs, trials_per_session, min_contrast, max_contrast, num_grading_levels):
    all_slope_diffs = []
    for i in range(all_model_outputs.shape[0]):
        slope_diffs = calculate_slope_differences_by_session(inputs, all_model_outputs[i], trials_per_session, min_contrast, max_contrast, num_grading_levels)
        all_slope_diffs.append(slope_diffs)
    
    return jnp.asarray(all_slope_diffs)


def calculate_conditional_bias_by_session(inputs, outputs, trials_per_session, specific_contrast_levels):
    _, _, rights_by_grading, totals_by_grading = generate_per_session_psychometrics(inputs, outputs, trials_per_session, 
                                                       num_grading_levels=NUM_GRADING_LEVELS,
                                                       specific_grading_levels=specific_contrast_levels)
    
    return np.sum(rights_by_grading, axis=1) / np.sum(totals_by_grading, axis=1)


def get_model_bias_evolution_from_outputs(true_outputs, all_model_outputs, trials_per_session, calculate_within_session=True):
    true_biases = get_biases_per_session(true_outputs, trials_per_session, calculate_within_session=calculate_within_session)
    model_biases = get_biases_per_session_batched(all_model_outputs, trials_per_session, calculate_within_session=calculate_within_session)
    variances = compute_variances_along_trajectory(model_biases)

    # plt.figure(figsize=(8,6))
    # plt.hist(model_ma_biases, bins=num_bins, color='skyblue', edgecolor='black')
    # plt.axvline(true_ma_bias, color='red', linestyle='dashed', linewidth=2, label='True Mouse')
    # plt.title('Histogram of Model-Agnostic Biases from Resampled Data', fontsize=16)
    # plt.xlabel('Bias', fontsize=14)
    # plt.ylabel('Frequency', fontsize=14)
    # if filepath is None:
    #     plt.show()
    # else:
    #     plt.savefig(f'{filepath}/ma_bias_distribution{sampling_tag}.png')

    return true_biases, model_biases, variances 



def plot_model_trace_evolution_by_session(true_traj, model_trajs, metric, sampling_type, 
                                          color='blue', title=None, filepath=None, fixed_grading=None, 
                                          custom_ticks=None, ymin=-1., ymax=1.):
    assert metric in ['Bias', 'Slope Difference']
    assert sampling_type in ['deterministic', 'MAP']
    sampling_tag = get_sampling_tag(sampling_type)

    fig, ax = plt.subplots(figsize=(6, 4))
    sessions = np.arange(1, len(true_traj)+1)
    num_sessions = len(sessions)
    for i in range(len(model_trajs)):
        ax.plot(sessions, model_trajs[i], color=color, lw=0.5, alpha=0.1, zorder=2)
    ax.plot(sessions, true_traj, color='red', lw=2.5, zorder=3)

    ax.axhline(0, color="#9E9D9D", linewidth=1., linestyle=':', zorder=1)
    
    x_upper = 5 * (num_sessions // 5)
    ax.set_xticks(np.arange(5, x_upper+1, 5))
    if custom_ticks is not None:
        ax.set_yticks(custom_ticks)
    else:
        ax.set_yticks([-1., -0.5, 0., 0.5, 1.])

    if ymax is not None:
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(-1., 1.)

    ax.set_xlabel('Session', fontsize=15)
    ax.set_ylabel(f'{metric}', fontsize=15)
    ax.set_title(f'{title}', fontsize=18)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.tick_params(labelsize=23, length=3, width=0.8)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    tag = ''
    if fixed_grading is not None:
        tag = f'{fixed_grading}_'
    
    if filepath is None:
        plt.show()
    else:
        plt.savefig(f'{filepath}/{metric.lower()}_{tag}evolution_by_session{sampling_tag}.pdf', bbox_inches='tight')
    return 


def plot_alpha_beta_scatterplot(dataset_name, alphas, betas, quantity, color_quantity, 
                                expert_mice, cmap, vmin, vmax, 
                                ylabel, color_label, nln_name, depth, figsize=(7, 4), 
                                ylims=(-0.05, 0.8), yticks=[0., 0.25, 0.5, 0.75, 1.0], filepath=None,
                                cticks=[0., 0.5, 1.0]):
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace': 0.35})

    norm = Normalize(vmin=vmin, vmax=vmax)

    for ax, vals, label, color in zip(
        axes,
        [alphas, betas],
        [r'$\alpha$ (learning rate)', r'$\beta$ (softmax gain)'],
        ["#D752D9", "#6C60F3"],
    ):
        if cmap is not None:
            ax.scatter(vals, quantity,
                c=color_quantity, cmap=cmap, norm=norm, s=100,
                linewidths=0.3, edgecolors='white', zorder=3)
        else:
            ax.scatter(vals, quantity, color=color, s=100,
                linewidths=0.3, edgecolors='white', zorder=3)
        
        ax.scatter(vals[expert_mice], quantity[expert_mice],
                   facecolors='none', edgecolors='black',
                   s=100, linewidths=1.2, zorder=4)

        ax.set_xscale('log')
        ax.set_yticks(yticks)
        # if nln_name == 'erf' and dataset_name == 'witten':
        #     ax.set_ylim(top=0.8)
        ax.set_ylim(ylims)
        ax.set_xlabel(label, fontsize=16)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.8)
        ax.tick_params(labelsize=16, length=3, width=0.8)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(f'{ylabel}', fontsize=16)
    axes[1].tick_params(labelleft=False)
    if dataset_name == 'witten':
        if nln_name == 'erf':
            axes[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
            axes[1].xaxis.set_minor_formatter(mticker.NullFormatter())
            axes[1].set_xticks([1, 2, 3, 4, 5])  
        if nln_name == 'linear' and depth == 'two-layer':
            axes[1].set_xticks([1, 10])
    
    # Shared colorbar
    if cmap is not None:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=1., pad=0.04, aspect=25)
        cbar.set_label(f'{color_label}', fontsize=16)
        cbar.set_ticks(cticks)
        cbar.ax.tick_params(labelsize=16, length=3, width=0.8)
        cbar.outline.set_linewidth(0.8)
    
    if filepath is not None:
        fig.savefig(f'{filepath}', dpi=300, bbox_inches='tight')
    else:
        plt.show()



def get_resampled_model_outputs(mouse_results_dict, inference_model, nln_name, filepath, key, 
                                num_test_sets=1_000, MAP_sampling=True):
    
    nln = get_nonlinearity(nln_name)
    inputs = mouse_results_dict['inputs']
    outputs = mouse_results_dict['outputs'].astype(jnp.int32)
    rewards = mouse_results_dict['rewards'] 
    targets = 1 - ((outputs + rewards.astype(jnp.int32)) % 2)
    targets = targets.reshape((-1, 1))
    
    optim_dict = mouse_results_dict['optim_dict']
    
    best_converged_hyperparams = mouse_results_dict['inferred_hyperparams']
    inferred_hyperparams, convergence_failed = find_best_hyperparams(optim_dict)
    
    if not convergence_failed:
        assert best_converged_hyperparams.all() == inferred_hyperparams.all()

    all_model_outputs = []

    w0_MAP, depth, lrs_MAP, beta_MAP = extract_fitted_params(mouse_results_dict, inference_model, return_dict=True)

    print(f'Depth is {depth}', flush=True)

    num_weights = NUM_TWO_LAYER_WEIGHTS
    if depth == 'one-layer':
        num_weights = NUM_ONE_LAYER_WEIGHTS

    skey, key = jr.split(key)
    
    # sample hyperparam vectors from the Laplace approximation to the posterior 
    x0 = inferred_hyperparams
    
    H, _ = compute_hessian(inferred_hyperparams, inputs, outputs, rewards, nln, inference_model, batched=False, alpha_W1=None, num_weights=num_weights, 
                           ridge_lambda=RIDGE_LAMBDA, alphabeta_ridge=RIDGE_ALPHABETA, mu_alpha=MU_ALPHA, mu_beta=MU_BETA)
    Sigma0 = jnp.linalg.inv(H)
    xs = jr.multivariate_normal(skey, x0, Sigma0, (num_test_sets,))
    sampling_tag = '_MAP_Laplace'
    
    # if Hessian has any eigenvalues that are negative or that blow up (for whatever reason), avoid MAP sampling
    if (not jnp.isfinite(xs).all()) or (not MAP_sampling):
        print(f"Proceeding without MAP sampling", flush=True)
        xs = jnp.tile(x0, (num_test_sets, 1))
        sampling_tag = ''
    
    print('Starting training process...', flush=True)

    for i in range(num_test_sets):
        _, key = jr.split(key)

        curr_hyperparams = xs[i]
        curr_lrs, curr_beta, curr_w0 = extract_lrs_beta_and_w0(curr_hyperparams, inference_model, return_dict=True)

        _, _, results_dict = train_gd_network(nln_name, curr_w0, inputs, targets, curr_lrs[0], curr_beta, 
                                              key, num_grading_levels=NUM_GRADING_LEVELS, print_stuff=False)

        model_outputs = results_dict['outputs']
        model_outputs = jnp.asarray(model_outputs)
        all_model_outputs.append(model_outputs)


    # make all_model_outputs a numpy array of dtype bool
    all_model_outputs = np.array(all_model_outputs).astype(bool) 
    if filepath is not None:
        np.save(f'{filepath}/all_model_outputs{sampling_tag}.npy', all_model_outputs)
        
        xs = np.asarray(xs)
        np.save(f'{filepath}/all_model_init_hyperparams{sampling_tag}.npy', xs)

    return all_model_outputs



def plot_single_trial_nlls(nll_data, mouse_ids, trial_idxs,
                           method_labels = {0: '1-layer GD', 1: '2-layer linear GD', 2: '2-layer erf GD'},
                           ylims=None, filepath=None, label='NLL', experts=False):
    assert label in ['NLL', 'AIC']
    A = nll_data    # num_trials x num_models
    num_test_items = A.shape[-1]
    num_trials = A.shape[0]

    baseline = np.tile(A[:, -1].reshape(-1, 1), (1, 3))  # 2-layer erf gd is "baseline" model -- relative NLL/AIC set to 0
    
    plot_label = f'Relative {label}'

    # Prepare data for seaborn
    records = []
    for trial in trial_idxs:
        curr_fip = mouse_ids[trial]
        for method in range(num_test_items):
            records.append({
                'Mouse': f'Mouse {curr_fip}',
                'Model Type': method_labels.get(method, f'Method {method}'),
                plot_label: A[trial, method] - baseline[trial, method]        
                })

    df = pd.DataFrame(records)

    # Plot: use stripplot instead of boxplot
    plt.figure(figsize=(13, 5))
    sns.stripplot(data=df, x='Mouse', y=plot_label, hue='Model Type', dodge=True, size=14, palette='tab10', jitter=False)

    plt.title(f"Single Trial Goodness of Fit", fontsize=20)
    plt.ylabel(plot_label)
    plt.xlabel("Mouse")
    plt.legend(title='Model Type')
    plt.tight_layout()

    expert_tag = ''
    if experts:
        expert_tag = 'experts_'

    if ylims is not None:
        plt.ylim(ylims)

    if filepath is None:
        plt.show()
    else:
        plt.savefig(f'{filepath}/{expert_tag}single_trial_{label}_comparison_trials{trial_idxs[0]}to{trial_idxs[-1]}.png')


def explore_expert_metrics(dataset_name, depth, nln_name, inference_model='gd', filepath=None,
                           plot_pc_idx1=1, plot_pc_idx2=2, final_window=500, remove_bad_idxs=True):
    
    assert dataset_name in ['witten', 'saxe', 'witten_and_saxe']

    if dataset_name != 'witten_and_saxe':
        all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=remove_bad_idxs)
        mouse_ids = list(all_mice_behav_dict.keys())
    else:
        all_mice_behav_dict_witten = load_data('witten', remove_bad_idxs=remove_bad_idxs)
        all_mice_behav_dict_saxe = load_data('saxe', remove_bad_idxs=remove_bad_idxs)
        all_mice_behav_dict = {**all_mice_behav_dict_witten, **all_mice_behav_dict_saxe}
        mouse_ids = list(all_mice_behav_dict.keys())

    n = len(mouse_ids)
    print(f'Number of mice: {n}')

    alphas = np.zeros(n)
    betas = np.zeros(n)
    hyperparams_collection = []
    ma_biases = np.zeros(n)

    avg_trial_weighted_accs = np.zeros(n)
    final_accs = np.zeros(n)

    expert_mice = []

    for i in range(n):
        mouse_id = mouse_ids[i]

        file_identifier = f'mouse_{mouse_id}'

        if dataset_name == 'witten_and_saxe':
            try:
                results_dict = pkl.load(open(f'witten_data_GD_analysis_final/fitted_models/{inference_model}_{nln_name}_{depth}/{file_identifier}_results_{depth}_{inference_model}_{nln_name}.pkl', 'rb'))
            except:
                results_dict = pkl.load(open(f'saxe_data_GD_analysis_final/fitted_models/{inference_model}_{nln_name}_{depth}/{file_identifier}_results_{depth}_{inference_model}_{nln_name}.pkl', 'rb'))
        else:
            results_dict = pkl.load(open(f'{dataset_name}_data_GD_analysis_final/fitted_models/{inference_model}_{nln_name}_{depth}/{file_identifier}_results_{depth}_{inference_model}_{nln_name}.pkl', 'rb'))

        inferred_hyperparams, _ = find_best_hyperparams(results_dict['optim_dict'])

        inputs = results_dict['inputs']
        rewards = results_dict['rewards']
        outputs = results_dict['outputs'].astype(jnp.int32)

        bias = calculate_simple_mouse_ma_bias(outputs)
        ma_biases[i] = bias

        expert = check_expert_status(all_mice_behav_dict[mouse_id])
        if expert:
            expert_mice.append(i)

        _, _, lrs, beta = extract_fitted_params(results_dict, inference_model)

        alphas[i] = lrs[0]
        betas[i] = beta
        hyperparams_collection.append(inferred_hyperparams)

        avg_trial_weighted_accs[i] = calculate_trial_averaged_acc(rewards, inputs)

        running_avg_acc = calculate_running_avg_quantity(rewards, window=final_window, mask=get_zero_contrast_mask(inputs))
        final_accs[i] = running_avg_acc[-1]
    
    print(f'Number of expert mice: {len(expert_mice)}')

    hyperparams_collection = np.array(hyperparams_collection)

    
    plot_alpha_beta_scatterplot(dataset_name, alphas, betas, final_accs, color_quantity=np.abs(ma_biases), 
                                expert_mice=expert_mice, cmap='YlOrRd', vmin=0., vmax=1., 
                                ylabel='Average Final Accuracy', color_label='|Bias|', nln_name=nln_name, depth=depth, figsize=(8, 4), 
                                ylims=(final_accs.min()-0.05, 1.), yticks=[0., 0.25, 0.5, 0.75, 1.0], filepath=f'{filepath}/alpha_beta_vs_final_acc.pdf')

    
    plot_alpha_beta_scatterplot(dataset_name, alphas, betas, avg_trial_weighted_accs, color_quantity=np.abs(ma_biases), 
                                expert_mice=expert_mice, cmap='YlOrRd', vmin=0., vmax=1., 
                                ylabel='Normalized Accuracy', color_label='|Bias|', nln_name=nln_name, depth=depth, 
                                figsize=(8, 4), ylims=(avg_trial_weighted_accs.min()-0.05, avg_trial_weighted_accs.max()+0.05), yticks=[-0.5, -0.25,0., 0.25, 0.5, 0.75, 1.0], filepath=f'{filepath}/alpha_beta_vs_avg_trial_weighted_normalized_accs.pdf')


    pca = PCA()
    pca.fit(hyperparams_collection)
    var_explained = pca.explained_variance_ratio_
    print('Proportion of variance explained by each PC:')
    print(var_explained)

    top_eigvecs = pca.components_[:3, :]
    plot_top_PCs(top_eigvecs, filepath=filepath)    

    pcs = pca.transform(hyperparams_collection)
    pc1 = pcs[:, 0]
    pc2 = pcs[:, 1]
    pc3 = pcs[:, 2]

    if nln_name == 'erf':
        plot_pc1 = pcs[:, plot_pc_idx1-1]
        plot_pc2 = pcs[:, plot_pc_idx2-1]


        c = np.sqrt(plot_pc1**2 + plot_pc2**2)

        plot_alpha_beta_scatterplot(dataset_name, alphas, betas, avg_trial_weighted_accs, color_quantity=c, 
                                    expert_mice=expert_mice, cmap='YlOrRd_r', vmin=0., vmax=c.max(), 
                                    ylabel='Normalized Accuracy', color_label=f'Magnitude of PCs {plot_pc_idx1} & {plot_pc_idx2}', nln_name=nln_name, depth=depth, 
                                    figsize=(8, 4), ylims=(avg_trial_weighted_accs.min()-0.05, avg_trial_weighted_accs.max()+0.05), yticks=[-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.0], filepath=f'{filepath}/all_hyperparams_vs_Normalized Accuracy.pdf', cticks=[0., 2., 4, 6., 8.])
        
        plot_alpha_beta_scatterplot(dataset_name, alphas, betas, np.abs(ma_biases), color_quantity=c, 
                                        expert_mice=expert_mice, cmap='YlOrRd_r', vmin=0., vmax=c.max(), 
                                        ylabel='|Bias|', color_label=f'Magnitude of PCs {plot_pc_idx1} & {plot_pc_idx2}', nln_name=nln_name, depth=depth, 
                                        figsize=(8, 4), ylims=(0., np.abs(ma_biases).max()+0.05), yticks=[0., 0.25, 0.5, 0.75, 1.0], filepath=f'{filepath}/all_hyperparams_vs_|Bias|.pdf', cticks=[0., 2., 4, 6., 8.])
        

        plot_linear_correlation(c, avg_trial_weighted_accs, r'$\sqrt{PC1^2 + PC2^2}$', f'Normalized \nAccuracy', filepath=filepath, figsize=(5, 5), labelsize=21, ticksize=20)
        plot_linear_correlation(c, np.abs(ma_biases), r'$\sqrt{PC1^2 + PC2^2}$', f'|Bias|', filepath=filepath, figsize=(5, 5), labelsize=21, ticksize=20)

        plot_PC_heatmap(plot_pc1, plot_pc2, plot_pc_idx1, plot_pc_idx2, avg_trial_weighted_accs, feature_name='avg_acc_score', filepath=filepath, cmap='YlOrRd_r')

        plot_PC_heatmap(plot_pc1, plot_pc2, plot_pc_idx1, plot_pc_idx2, np.abs(ma_biases), feature_name='ma_bias', filepath=filepath, cmap='YlOrRd')

    if dataset_name == 'saxe' and nln_name == 'erf':
        # check for correlations of acc and bias with initial first-layer stimulus weights
        plot_linear_correlation(np.abs(hyperparams_collection[:, 3]), avg_trial_weighted_accs, 'Layer-1 Right', f'Normalized \nAccuracy', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)
        plot_linear_correlation(np.abs(hyperparams_collection[:, 4]), avg_trial_weighted_accs, 'Layer-1 Left', f'Normalized \nAccuracy', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)
        
        plot_linear_correlation(np.abs(hyperparams_collection[:, 3]), np.abs(ma_biases), 'Layer-1 Right', '|Bias|', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)
        plot_linear_correlation(np.abs(hyperparams_collection[:, 4]), np.abs(ma_biases), 'Layer-1 Left', '|Bias|', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)

        layer_one_strength = np.sqrt(hyperparams_collection[:, 3]**2 + hyperparams_collection[:, 4]**2)
        plot_alpha_beta_scatterplot(dataset_name, alphas, betas, avg_trial_weighted_accs, color_quantity=layer_one_strength, 
                                    expert_mice=expert_mice, cmap='YlOrRd_r', vmin=0., vmax=layer_one_strength.max(), 
                                    ylabel='Normalized Accuracy', color_label=f'Magnitude of Layer-1 Stim Weights', nln_name=nln_name, depth=depth, 
                                    figsize=(8, 4), ylims=(avg_trial_weighted_accs.min()-0.05, avg_trial_weighted_accs.max()+0.05), yticks=[-0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.0], filepath=f'{filepath}/layer_one_and_alpha_and_beta_vs_Normalized Accuracy.pdf', cticks=[0., 2., 4, 6., 8.])


    plot_linear_correlation(np.abs(pc1), avg_trial_weighted_accs, '|PC 1|', f'Normalized \nAccuracy', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)
    plot_linear_correlation(np.abs(pc2), avg_trial_weighted_accs, '|PC 2|', f'Normalized \nAccuracy', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)
    plot_linear_correlation(np.abs(pc3), avg_trial_weighted_accs, '|PC 3|', f'Normalized \nAccuracy', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)

    plot_linear_correlation(np.abs(ma_biases), avg_trial_weighted_accs, '|Bias|', 'Normalized Accuracy', filepath=filepath, labelsize=21, ticksize=20, figsize=(5, 5))

    plot_linear_correlation(np.abs(pc1), np.abs(ma_biases), '|PC 1|', '|Bias|', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)
    plot_linear_correlation(np.abs(pc2), np.abs(ma_biases), '|PC 2|', '|Bias|', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)
    plot_linear_correlation(np.abs(pc3), np.abs(ma_biases), '|PC 3|', '|Bias|', filepath=filepath, figsize=(6, 3), labelsize=24, ticksize=22)



def generate_aggregate_model_metrics(dataset_name, plot_pc_idx1=1, plot_pc_idx2=2, remove_bad_idxs=True):
    erf_directory = f'{dataset_name}_data_GD_analysis_final/aggregate_results/gd_erf_two-layer'
    explore_expert_metrics(dataset_name, depth='two-layer', nln_name='erf', inference_model='gd', filepath=erf_directory,
                           plot_pc_idx1=plot_pc_idx1, plot_pc_idx2=plot_pc_idx2, remove_bad_idxs=remove_bad_idxs)
    
    print('Done with 2-layer nonlinear, starting 2-layer linear...')
    explore_expert_metrics(dataset_name, depth='two-layer', nln_name='linear', inference_model='gd', filepath=f'{dataset_name}_data_GD_analysis_final/aggregate_results/gd_linear_two-layer',
                           plot_pc_idx1=plot_pc_idx1, plot_pc_idx2=plot_pc_idx2)
    
    print('Done with 2-layer linear, starting 1-layer linear...')
    explore_expert_metrics(dataset_name, depth='one-layer', nln_name='linear', inference_model='gd', filepath=f'{dataset_name}_data_GD_analysis_final/aggregate_results/gd_linear_one-layer',
                           plot_pc_idx1=plot_pc_idx1, plot_pc_idx2=plot_pc_idx2)


def make_single_mouse_aic_plot(dataset_name, mouse_id, save_path=None):
    MODEL_COLORS = ['#2166AC', '#F4A736', '#2CA02C']
    MODEL_NAMES  = ['1-layer\nLinear', '2-layer\nLinear', '2-layer\nNonlinear']

    aics, _ = get_mouse_aics(dataset_name, mouse_id)

    fig, ax = plt.subplots(figsize=(4, 3))

    for i, (val, color, name) in enumerate(zip(aics, MODEL_COLORS, MODEL_NAMES)):
        ax.scatter(i+0.5, val, color=color, s=80, zorder=4, clip_on=False)
        ax.vlines(i+0.5, min(aics) * 0.98, val,
                  colors=color, linewidth=1.2, alpha=0.4, zorder=3)

    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(MODEL_NAMES, fontsize=14)
    ax.set_ylabel('AIC', fontsize=16)
    ax.set_ylim(bottom=min(aics) * 0.98)
    ax.set_xlim(left=0, right=3)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.tick_params(axis='y', labelsize=14, length=3, width=0.8)
    ax.tick_params(axis='x', length=0)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def generate_model_nll_comparison(dataset_name, get_aic_per_trial=False, remove_bad_idxs=True):
    assert dataset_name in ['witten', 'saxe']
    nll_data = []
    aic_data = []

    expert_idxs = []
    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=remove_bad_idxs)
    mouse_ids = list(all_mice_behav_dict.keys())

    for i in range(len(mouse_ids)):
        mouse_id = mouse_ids[i]

        num_trials = len(all_mice_behav_dict[mouse_id]['rewards'])

        expert = check_expert_status(all_mice_behav_dict[mouse_id], threshold=EXPERT_THRESHOLD, num_final_sessions=NUM_FINAL_SESSIONS)
        if expert:
            expert_idxs.append(i)

        mouse_aics, mouse_nlls = get_mouse_aics(dataset_name, mouse_id)
        mouse_aics, mouse_nlls = np.asarray(mouse_aics), np.asarray(mouse_nlls)
        if get_aic_per_trial:
            mouse_aics /= num_trials 
            mouse_nlls /= num_trials

        nll_data.append(mouse_nlls)
        aic_data.append(mouse_aics)
    
    nll_data = np.array(nll_data)   # num_mice x num_models
    aic_data = np.array(aic_data)
    expert_idxs = np.array(expert_idxs)

    # find argmin across models for each mouse
    min_idxs = np.argmin(aic_data, axis=1)
    print(min_idxs)
    # find number of idxs that are '2'
    num_2nl_winners = np.sum(min_idxs == 2)
    total_idxs = len(min_idxs)
    print(f'Proportion of mice best fit by 2-layer nonlinear model: {num_2nl_winners}/{total_idxs} = {num_2nl_winners/total_idxs:.2f}')
    
    expert_aic_data = aic_data[expert_idxs]
    min_expert_idxs = np.argmin(expert_aic_data, axis=1)
    print(min_expert_idxs)

    return aic_data, nll_data, expert_idxs


def find_winners(aic_data, idx1, idx2):
    winners = []
    for i in range(len(aic_data)):
        if aic_data[i, idx1] <= aic_data[i, idx2]:
            winners.append(idx1)
        else:
            winners.append(idx2)
    
    return winners


def make_jitterplot(data, expert_idxs, save_path=None, 
                    plot_type='aic_per_trial', ylabel=r'$\Delta$AIC', show_median=False, 
                    plot_sem=False, plot_ordering=[0, 1, 2], show_legend=False, ticksize=19,
                    custom_yticks=None):
    assert plot_type in ['aic_per_trial', 'r2']
    MODEL_COLORS = ['#2166AC', '#F4A736', '#2CA02C']  # blue, orange, green
    MODEL_NAMES  = ['1-layer Linear', '2-layer Linear', '2-layer Nonlinear']
    MODEL_TO_IDX = {'one-layer linear gd': 0, 'two-layer linear gd': 1, 'two-layer erf gd': 2}
    EXPERT_MEAN_COLOR   = "#000000"  # near-black
    NONEXPERT_MEAN_COLOR = "#A29E9E"  # mid-gray

    expert_idxs = np.asarray(expert_idxs, dtype=int)

    bar_width = 0.3  # half-width of the mean bar
    non_expert_idxs = np.array([i for i in range(len(data)) if i not in expert_idxs])

    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    one_layer_diffs = xs - zs
    two_layer_diffs = ys - zs
    linear_diffs = xs - ys

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 6.5), sharey=True)

    axes[0].set_ylabel(f'{ylabel}', fontsize=19)
    
    if plot_type == 'aic_per_trial':
        axes[0].set_yticks([0, 0.15, 0.3])
    if plot_type == 'r2':
        axes[0].set_yticks([-2., 0, 2.])
        plt.ylim(bottom=-3.2, top=3.2)
    
    if custom_yticks is not None:
        axes[0].set_yticks(custom_yticks)

    for idxs, plot_idx, diffs in zip([[0, 2], [1, 2], [0, 1]], plot_ordering,
                                 [one_layer_diffs, two_layer_diffs, linear_diffs]):
        jitter = rng.uniform(-0.25, 0.25, size=len(diffs))

        sign = 1.
        if plot_type == 'r2':
            sign = -1.
        winners = find_winners(sign * data, idxs[0], idxs[1])
        colors = [MODEL_COLORS[w] for w in winners]

        axes[plot_idx].scatter(jitter, diffs, c=colors, s=200,
                   edgecolors='white', linewidths=0.3, zorder=3)
        axes[plot_idx].scatter(jitter[expert_idxs], diffs[expert_idxs],
                   facecolors='none', edgecolors='black',
                   s=200, linewidths=1.2, zorder=4)

        axes[plot_idx].axhline(0, color='lightgray', linestyle='--', linewidth=0.9, zorder=2)
        axes[plot_idx].set_xticks([])
        axes[plot_idx].spines[['top', 'right', 'bottom']].set_visible(False)
        axes[plot_idx].spines['left'].set_linewidth(0.8)
        axes[plot_idx].tick_params(labelsize=ticksize, length=3, width=0.8)
        axes[plot_idx].yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
        axes[plot_idx].set_axisbelow(True)

        for mean, median, sem, color, linestyle in [
            (diffs[non_expert_idxs].mean(), np.median(diffs[non_expert_idxs]), diffs[non_expert_idxs].std() / np.sqrt(len(non_expert_idxs)), NONEXPERT_MEAN_COLOR, 'solid'),
            (diffs[expert_idxs].mean(), np.median(diffs[expert_idxs]), diffs[expert_idxs].std() / np.sqrt(len(expert_idxs)), EXPERT_MEAN_COLOR, 'solid'),
        ]:
            if show_median:
                axes[plot_idx].hlines(median, -bar_width, bar_width,
                    colors=color, linewidth=1.5, zorder=5, linestyle=linestyle)
            else:
                axes[plot_idx].hlines(mean, -bar_width, bar_width,
                        colors=color, linewidth=1.5, zorder=5, linestyle=linestyle)
                
                if plot_sem:
                    axes[plot_idx].vlines(0, mean - sem, mean + sem,
                        colors=color, linewidth=1.5, zorder=5)

    if show_legend:
        legend_handles = [mpatches.Patch(color=c, label=n)
                        for c, n in zip(MODEL_COLORS, MODEL_NAMES)]
        legend_handles += [Line2D([0], [0], marker='o', color='none', ms=7,
                                markeredgecolor='black', markeredgewidth=1.2, label='Expert')]

        fig.legend(
        handles=legend_handles,
        fontsize=17, frameon=False,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),  # was 0.93, now sits just inside top of figure
        ncol=2,
        )

        plt.tight_layout(rect=[0, 0.15, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def make_2D_scatterplot(data, model1_name, model2_name, expert_idxs, save_path=None, 
                            title='', measure=r'$R^2$'):
    assert measure in [r'$R^2$', 'AIC']
    MODEL_COLORS = ['#2166AC', '#F4A736', '#2CA02C']  # blue, orange, green
    MODEL_NAMES  = ['1-Layer Linear', '2-Layer Linear', '2-Layer Nonlinear']
    MODEL_TO_IDX = {'one-layer linear': 0, 'two-layer linear': 1, 'two-layer nonlinear': 2}
    sign = 1.
    if measure == r'$R^2$':
        sign = -1.

    assert model1_name in list(MODEL_TO_IDX.keys())
    assert model2_name in list(MODEL_TO_IDX.keys())

    idx1 = MODEL_TO_IDX[model1_name]
    idx2 = MODEL_TO_IDX[model2_name]

    # winner = np.argmin(aic_data, axis=1)  # 0, 1, or 2 per mouse
    winner = []
    for i in range(len(data)):
        if sign * data[i, idx1] <= sign * data[i, idx2]:
            winner.append(idx1)
        else:
            winner.append(idx2)

    colors = [MODEL_COLORS[w] for w in winner]

    xs = data[:, idx1] 
    ys = data[:, idx2] 

    for i in range(len(xs)):
        if xs[i] < 0 or ys[i] < 0:
            print((xs[i], ys[i]))

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    xs_plot = np.maximum(xs, 0)
    ys_plot = np.maximum(ys, 0)

    # handle origin separately with jitter to avoid fully plotting over a previous point at the origin
    rng = np.random.default_rng(100)
    jitter = 0.02  # tune to your axis scale
    at_origin = (xs_plot == 0) & (ys_plot == 0)
    xs_plot[at_origin] += rng.uniform(0, jitter, at_origin.sum())
    ys_plot[at_origin] += rng.uniform(0, jitter, at_origin.sum())

    ax.scatter(xs_plot, ys_plot, c=colors, s=125, alpha=0.8, linewidths=0.3, edgecolors='white', zorder=3)

    ax.scatter(xs_plot[expert_idxs], ys_plot[expert_idxs],
           facecolors='none', edgecolors="#000000",
           s=125, linewidths=1.2, zorder=4)

    # Reference lines
    # lim = max(np.abs([xs.min(), xs.max(), ys.min(), ys.max()])) * 1.1
    # ax.axhline(0, color='gray', linestyle='--', linewidth=0.9, zorder=2)
    # ax.axvline(0, color='gray', linestyle='--', linewidth=0.9, zorder=2)

    # ax.plot([-lim, lim], [-lim, lim], color='#888888', linestyle=':', linewidth=0.9, zorder=2)
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    if measure == 'AIC':
        # pad = 200
        # x_lo, x_hi = np.percentile(xs, 2) - pad, np.percentile(xs, 98) + pad
        # y_lo, y_hi = np.percentile(ys, 2) - pad, np.percentile(ys, 98) + pad

        # # Make square by taking the union of both ranges
        # lo = min(x_lo, y_lo)
        # hi = max(x_hi, y_hi)
        lo = 0. 
        hi = max(np.max(xs), np.max(ys)) * 1.1
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        # Recompute the y=x line over the new range
        ax.plot([lo, hi], [lo, hi], color='#888888', linestyle=':', linewidth=1.3, zorder=2)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        # plot y = x
        ax.plot([0, 1], [0, 1], color='#888888', linestyle=':', linewidth=1.3, zorder=2)


    ax.set_xlabel(f'{MODEL_NAMES[idx1]}', fontsize=19)
    ax.set_ylabel(f'{MODEL_NAMES[idx2]}', fontsize=19)
    ax.set_title(f'{title}', fontsize=14)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.tick_params(labelsize=18, length=3, width=0.8)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.xaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)
    ax.set_aspect('equal')

    # legend_handles = [mpatches.Patch(color=MODEL_COLORS[i], label=MODEL_NAMES[i]) for i in [idx1, idx2]]

    # legend_handles += [Line2D([0], [0], marker='o', color='none', ms=7,
    #    markeredgecolor="#000000", markeredgewidth=1.2, label='Expert')]

    # # ax.legend(handles=legend_handles, fontsize=13, frameon=False,
    # #       loc='upper left', title='Best Model', title_fontsize=13)
    # ax.legend(
    #     handles=legend_handles,
    #     fontsize=13,
    #     frameon=False,
    #     loc='lower center',
    #     bbox_to_anchor=(0.5, 1.02),   # centered just above the axes
    #     ncols=len(legend_handles),    # single row
    #     title='Best Model',
    #     title_fontsize=13,
    # )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def get_stage_center(stage, length):
    assert stage in ['all', 'early', 'middle', 'late']
    if stage in ['all', 'early']:
        return 0
    elif stage == 'late':
        return length
    else:
        return int(length/2)


def get_stage_bounds(num_time_steps, stage, length=2_000, true_mouse_accs=None, window=None):
    assert stage in ['all', 'early', 'middle', 'late']
    xmin, xmax = None, None
    if stage == 'early':
        xmin = 0
        xmax = length
    elif stage == 'late':
        xmin = num_time_steps - length 
        xmax = num_time_steps
    elif stage == 'middle':
        assert true_mouse_accs is not None 
        assert window is not None 
        xs = np.arange(window-1, len(true_mouse_accs))
        y_fit, (L, x0, k, b) = get_sigmoid_fit(xs, true_mouse_accs[window-1:])
        xmin = int(x0 - length/2)
        xmax = int(x0 + length/2)
        if xmin < 0:
            xmin = 0
            xmax = length
        if xmax > num_time_steps:
            xmin = num_time_steps - length
            xmax = num_time_steps
    else:
        xmin = 0
        xmax = num_time_steps
    return xmin, xmax


def get_stage_accs(dataset_name, mouse_id, stage, stage_length, top_k, window, stride, 
                   sampling_type='deterministic', trials_per_session=None, plot=True, save_label='', save=False, ymin=0.4):
    assert stage in ['early', 'middle', 'late']
    plot_filepath = f'{dataset_name}_data_GD_analysis_final/single_trial_results/running_accs/single_trial'

    mean_accs = {'true_mouse': np.zeros(stage_length), '1-Layer Linear': np.zeros(stage_length),
                 '2-Layer Linear': np.zeros(stage_length), '2-Layer Nonlinear': np.zeros(stage_length)}
    
    # xcenter = get_stage_center(stage, stage_length)
    
    running_accs_info, _, _, _ = get_reconstructed_running_accs(dataset_name, mouse_id,
                                        top_k, window=window, stride=stride, sampling_type=sampling_type,
                                        plot=False, legend=False, save=False)
    true_mouse_accs = running_accs_info['true_mouse']

    xmin, xmax = get_stage_bounds(len(true_mouse_accs), stage, length=stage_length, 
                                    true_mouse_accs=true_mouse_accs, window=window)
    
    # xs = np.arange(stage_length) - xcenter + 1
    xs = np.arange(xmin+1, xmax+1, dtype=int)
    
    print(mouse_id)
    
    mean_accs['true_mouse'] = true_mouse_accs[xmin:xmax]  
    for key in MODEL_CLASS_INFO:
        mean_accs[key] = running_accs_info[key]['avg'][xmin:xmax]  
    
    if plot:
        fig, ax = plt.subplots(figsize=(4.5, 3))

        ax.plot(xs[::stride], mean_accs['true_mouse'][::stride],
                label='True Mouse', color='#D62728', lw=2.0, zorder=4)

        for key, value in mean_accs.items():
            if key == 'true_mouse':
                continue
            color = MODEL_CLASS_INFO[key][4]
            model_accs = value
            ax.plot(xs[::stride], model_accs[::stride], color=color, lw=1.5,
                    linestyle='--', label=key, zorder=3)
            
        if trials_per_session is not None:
            session_bounds = np.cumsum(trials_per_session)
            for bound in session_bounds:
                if bound > xmin and bound < xmax:
                    ax.axvline(bound, color="#777575", linewidth=1.5, linestyle='dashed', zorder=2)

        ax.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', zorder=1)

        ax.set_xlabel('Trial', fontsize=19)
        ax.set_ylabel('Running Accuracy', fontsize=19)
        # ax.set_title(f'', fontsize=22)
        ax.set_ylim(ymin, 1.0)
        ax.set_yticks([0.5, 0.75, 1.0])
        
        ax.locator_params(axis='x', nbins=3)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax.set_xlim(np.maximum(xmin, window)-1, xmax+1)

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.8)
        ax.tick_params(labelsize=17, length=3, width=0.8)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
        ax.set_axisbelow(True)
        # ax.legend(fontsize=8.5, frameon=False)

        
        plt.tight_layout()
        if save:
            plt.savefig(f'{plot_filepath}/avg_running_accs_{save_label}_{stage}_stage_length={stage_length}_top{top_k}_window={window}_stride={stride}.pdf',
                        dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    return


def get_pearson_r2_for_accs(dataset_name, mouse_id, stage, stage_length, top_k, window, 
                            sampling_type='deterministic', remove_bad_idxs=True):
    assert (dataset_name == 'witten') or (dataset_name == 'saxe')
    assert sampling_type in ['MAP', 'deterministic']

    sampling_tag = get_sampling_tag(sampling_type)
    file_identifier = f'mouse_{mouse_id}'

    choice_keyword = DATASET_TO_CHOICE_KEYWORD[dataset_name]
    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=remove_bad_idxs)
    
    mouse_behav_dict = all_mice_behav_dict[mouse_id]

    true_outputs = np.asarray(mouse_behav_dict[choice_keyword])
    true_outputs = true_outputs.astype(np.int32)

    true_rewards = np.asarray(mouse_behav_dict['rewards'])
    targets = 1 - ((true_outputs + true_rewards.astype(jnp.int32)) % 2)

    true_mouse_accs = calculate_running_avg_quantity(true_rewards, window)

    xmin, xmax = get_stage_bounds(len(true_outputs), stage, length=stage_length, 
                                  true_mouse_accs=true_mouse_accs, window=window)

    pearson_dists = []

    for key, value in MODEL_CLASS_INFO.items():
        inference_model, depth, nln_name, _, color = value

        all_model_outputs = np.load(f'{dataset_name}_data_GD_analysis_final/single_trial_results/{inference_model}_{nln_name}_{depth}_results/{file_identifier}/all_model_outputs{sampling_tag}.npy')

        _, k_best_model_outputs, _ = find_best_matching_model_outputs(true_outputs, all_model_outputs, top_k=top_k)
        k_best_rewards = (k_best_model_outputs == targets.reshape(1, -1)).astype(jnp.int32)

        all_model_running_accs = calculate_running_avg_quantity(k_best_rewards, window)
        pearson_rs = get_pearson_correlations(all_model_running_accs[:, xmin:xmax], true_mouse_accs[xmin:xmax])

        pearson_dists.append(pearson_rs)

    return pearson_dists


def get_reconstructed_running_accs(dataset_name, mouse_id,
                                    top_k, window, stride, sampling_type='deterministic', 
                                    xmin=0, xmax=None, plot=True, legend=False, save=True,
                                    plot_filepath=None, custom_xticks=None, error_bars='minmax',
                                    remove_bad_idxs=True):
    assert (dataset_name == 'witten') or (dataset_name == 'saxe')
    assert sampling_type in ['MAP', 'deterministic']
    assert error_bars in ['minmax', 'sem', 'std']
    if plot and save:
        assert plot_filepath is not None

    sampling_tag = get_sampling_tag(sampling_type)
    file_identifier = f'mouse_{mouse_id}'

    choice_keyword = DATASET_TO_CHOICE_KEYWORD[dataset_name]
    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=remove_bad_idxs)
    
    mouse_behav_dict = all_mice_behav_dict[mouse_id]

    true_outputs = np.asarray(mouse_behav_dict[choice_keyword])
    true_outputs = true_outputs.astype(np.int32)

    true_rewards = np.asarray(mouse_behav_dict['rewards'])
    targets = 1 - ((true_outputs + true_rewards.astype(jnp.int32)) % 2)

    true_running_accs = calculate_running_avg_quantity(true_rewards, window)

    # plot_filepath = f'{dataset_name}_data_GD_analysis_final/single_trial_results/running_accs'

    num_trials = len(targets)
    # colors = ['blue', 'orange', 'green']

    running_accs_info = {'true_mouse': true_running_accs}
    r2s = []
    mses = []
    l1_output_errors = []

    i = 0
    for key, value in MODEL_CLASS_INFO.items():
        inference_model, depth, nln_name, _, color = value

        all_model_outputs = np.load(f'{dataset_name}_data_GD_analysis_final/single_trial_results/{inference_model}_{nln_name}_{depth}_results/{file_identifier}/all_model_outputs{sampling_tag}.npy')

        _, k_best_model_outputs, _ = find_best_matching_model_outputs(true_outputs, all_model_outputs, top_k=top_k)
        avg_running_accs, min_running_accs, max_running_accs, std_running_accs, sem_running_accs = get_running_accs_from_best_matching_outputs(true_outputs, all_model_outputs, targets, top_k, window)

        running_accs_info[MODEL_PLOT_NAMES[i]] = {'avg': avg_running_accs, 'min': min_running_accs, 'max': max_running_accs, 
                                                  'std': std_running_accs, 'sem': sem_running_accs, 'color': color}

        r_squared = compute_coeff_of_determination(avg_running_accs, true_running_accs)
        mse = np.nanmean((avg_running_accs - true_running_accs)**2)
        # print(f'{key}: {r_squared}')

        l1_errors = jnp.sum(jnp.abs(k_best_model_outputs - true_outputs.reshape(1, -1)), axis=1) / len(true_outputs)
        l1_output_errors.append(l1_errors)

        r2s.append(r_squared)
        mses.append(mse)
        i += 1


    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        if xmax is None:
            xmax = num_trials
        trials = np.arange(xmin+1, xmax+1)[::stride]

        ax.plot(trials, true_running_accs[xmin:xmax:stride],
                label='True Mouse', color='#D62728', lw=2.0, zorder=4)

        for key, value in running_accs_info.items():
            if key == 'true_mouse':
                continue
            color = value['color']
            avg = value['avg']
            lo, hi = None, None 
            if error_bars == 'minmax':
                lo, hi = value['min'], value['max']
            elif error_bars == 'sem':
                lo, hi = value['avg'] - value['sem'], value['avg'] + value['sem']
            else:
                lo, hi = value['avg'] - value['std'], value['avg'] + value['std']
            ax.plot(trials, avg[xmin:xmax:stride], color=color, lw=1.5,
                    linestyle='--', label=key, zorder=3)
            ax.fill_between(trials, lo[xmin:xmax:stride], hi[xmin:xmax:stride],
                            color=color, alpha=0.12, linewidth=0, zorder=2)

        ax.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', zorder=1)

        ax.set_xlabel('Trial', fontsize=24)
        ax.set_ylabel('Running Accuracy', fontsize=19)
        ax.set_title(f'Mouse {mouse_id}', fontsize=25)
        ax.set_ylim(0.3, 1.0)
        ax.set_yticks([0.5, 0.75, 1.0])

        if xmax == num_trials and xmin == 0:
            upper_x = num_trials 
            xticks = np.arange(5000, upper_x+1, 5000)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f'{n:,}' for n in xticks], fontsize=13)

        if custom_xticks is not None:
            ax.set_xticks(custom_xticks)
            ax.set_xlim(left=window-5)

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.8)
        ax.tick_params(labelsize=22, length=3, width=0.8)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
        ax.set_axisbelow(True)
        # ax.legend(fontsize=8.5, frameon=False)

        label = ''
        if xmax is not None:
            label = f'_xmin={xmin}_xmax={xmax}'

        plt.tight_layout()
        if save:
            plt.savefig(f'{plot_filepath}/{file_identifier}_top{top_k}_avg_running_accs_window={window}_stride={stride}_error_{error_bars}_{label}.pdf',
                        dpi=300, bbox_inches='tight')
        else:
            plt.show()

        if legend:
            handles, labels = ax.get_legend_handles_labels()
            fig_legend, ax_legend = plt.subplots(figsize=(3, 1.5))
            ax_legend.legend(handles, labels, fontsize=8.5, frameon=False, loc='center')
            ax_legend.axis('off')
            fig_legend.savefig(f'{plot_filepath}/legend.pdf', dpi=300, bbox_inches='tight')
            # plt.show()


    return running_accs_info, r2s, mses, np.asarray(l1_output_errors)


def plot_acc_alignment_distributions(dataset_name, mouse_ids,
                                     top_k, window, stride, num_bins, 
                                     sampling_type='MAP', filepath=None):
    MODEL_COLORS = ['#2166AC', '#F4A736', '#2CA02C']
    
    r2s = []
    mses = []

    for mouse_id in mouse_ids:
        _, curr_r2s, curr_mses, _ = get_reconstructed_running_accs(dataset_name, mouse_id, top_k, window, 
                                                                 stride, sampling_type=sampling_type, plot=False)
        
        r2s.append(curr_r2s)
        mses.append(curr_mses)
    
    r2s = np.asarray(r2s)
    mses = np.asarray(mses)

    mean_r2 = r2s.mean(axis=0)
    mean_mse = mses.mean(axis=0)

    print(f'Mean R^2: {mean_r2}')
    print(f'Mean MSE: {mean_mse}')

    print(r2s)

    # plt.figure()
    # for i, (key, value) in enumerate(MODEL_CLASS_INFO.items()):
    #     plt.hist(r2s[:, i], num_bins, density=True, label=key, alpha=0.5)
    # plt.xlabel('$R^2$')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # for i, (key, value) in enumerate(MODEL_CLASS_INFO.items()):
    #     plt.hist(mses[:, i], num_bins, density=True, label=key, alpha=0.5)
    # plt.xlabel('MSE')
    # plt.legend()
    # plt.show()

    for data, xlabel, xlim, fname in [
        (r2s, '$R^2$ (Coeff. of Determination)', (0, 1), 'r2_histogram'),
        # (mses, 'MSE', (0, 0.03), 'mse_histogram'),
    ]:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))

        # bins = np.linspace(xlim[0], xlim[1], num_bins + 1)
        global_min = data.min()  # or data[:, i].min() per-distribution
        bins = np.linspace(global_min, xlim[1], num_bins + 1)

        for i, (name, color) in enumerate(zip(list(MODEL_CLASS_INFO.keys()), MODEL_COLORS)):
            ax.hist(data[:, i], bins=bins, density=False,
                    label=name, color=color, alpha=0.35,
                    edgecolor='white', linewidth=0.4)

        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel('Count', fontsize=15)
        ax.set_xlim(xlim)
        ax.set_yticks([0, 3, 6])

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.8)
        ax.tick_params(labelsize=13.5, length=3, width=0.8)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=12, frameon=False)

        plt.tight_layout()
        if filepath is not None:
            plt.savefig(f'{filepath}', dpi=300, bbox_inches='tight')
        plt.show()

    return mean_r2 #, mean_mse
    

def plot_l1_distances(l1_data, expert_idxs=None, filepath=None,
                      plot_type='all'):
    """
    l1_data : array of shape (num_mice, num_models, 10)
    """

    assert plot_type in ['all', 'expert', 'non-expert']
    if plot_type == 'expert':
        l1_data = l1_data[expert_idxs]
    if plot_type == 'non-expert':
        nonexpert_idxs = [idx for idx in range(len(l1_data)) if idx not in expert_idxs]
        l1_data = l1_data[nonexpert_idxs]

    MODEL_NAMES  = ['1-Layer\nLinear', '2-Layer\nLinear', '2-Layer\nNonlinear']

    # Per-mouse mean L1 across the 10 retrains
    mouse_means = l1_data.mean(axis=2)   # (num_mice, num_models)
    n_mice, n_models = mouse_means.shape
    xs = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(7, 5))

    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=(n_mice,))

    # Connecting lines (one per mouse)
    for i in range(n_mice):
        ax.plot(xs + jitter[i], mouse_means[i], color='gray', lw=0.6, alpha=0.3, zorder=1)

    # Dots per mouse per model
    for j, (color, name) in enumerate(zip(MODEL_COLORS, MODEL_NAMES)):
        ax.scatter(xs[j] + jitter, mouse_means[:, j],
                   color=color, s=22, alpha=1.0,
                   edgecolors='white', linewidths=0.3, zorder=3)
        if plot_type == 'expert':
            ax.scatter(xs[j] + jitter, mouse_means[:, j],
                       facecolors='none', edgecolors='black',
                       s=22, linewidths=1.0, zorder=5)

    # Mean ± SEM bar per model
    for j, color in enumerate(MODEL_COLORS):
        mean = mouse_means[:, j].mean()
        sem  = mouse_means[:, j].std() / np.sqrt(n_mice)
        ax.hlines(mean, xs[j] - 0.2, xs[j] + 0.2,
                  colors=color, linewidth=2.5, zorder=4)
        ax.vlines(xs[j], mean - sem, mean + sem,
                  colors=color, linewidth=1.5, zorder=4)

    # Expert outlines
    if plot_type == 'all' and expert_idxs is not None:
        for j in range(n_models):
            ax.scatter(xs[j] + jitter[expert_idxs], mouse_means[expert_idxs, j],
                       facecolors='none', edgecolors='black',
                       s=22, linewidths=1.0, zorder=5)

    ax.set_xticks(xs)
    ax.set_xticklabels(MODEL_NAMES, fontsize=19.5)
    ax.set_ylabel('L1 Distance', fontsize=19.5)
    ax.set_xlim(-0.5, n_models - 0.5)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.tick_params(labelsize=18, length=3, width=0.8)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if filepath is not None:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()



def plot_bias_distribution_results(dataset_name, mouse_id, sampling_type, num_bins=30, remove_bad_idxs=True):

    file_identifier = f'mouse_{mouse_id}'
    sampling_tag = get_sampling_tag(sampling_type)

    choice_keyword = DATASET_TO_CHOICE_KEYWORD[dataset_name]
    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=remove_bad_idxs)

    mouse_behav_dict = all_mice_behav_dict[mouse_id]
    trials_per_session = mouse_behav_dict['trials_per_session']
    
    true_outputs = np.asarray(mouse_behav_dict[choice_keyword])
    true_outputs = true_outputs.astype(np.int32)

    full_filepath = f'{dataset_name}_data_GD_analysis_final/single_trial_results/{inference_model}_{nln_name}_{depth}_results/{file_identifier}'

    all_model_outputs = np.load(f'{full_filepath}/all_model_outputs{sampling_tag}.npy')

    get_model_bias_evolution_from_outputs(true_outputs, all_model_outputs, trials_per_session, full_filepath, mouse_id, num_bins=num_bins)


def visualize_single_mouse_model_weights_and_aics(dataset_name, remove_bad_idxs=True):
    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=remove_bad_idxs)
    mouse_ids = list(all_mice_behav_dict.keys())

    for mouse_id in mouse_ids:
        aic_save_path = f'{dataset_name}_data_GD_analysis_final/single_trial_results/single_mouse_aics/mouse_{mouse_id}_aics.pdf'
        weighttraj_save_path = f'{dataset_name}_data_GD_analysis_final/single_trial_results/mouse_inferred_weights/mouse_{mouse_id}_inferred_weights.pdf'

        visualize_mouse_inferred_weights(dataset_name, mouse_id, save_path=weighttraj_save_path, stride=1)
        make_single_mouse_aic_plot(dataset_name, mouse_id, save_path=aic_save_path)


