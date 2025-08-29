import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
from jax.tree_util import tree_map
import optax
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


def generate_simulated_binary_data(num_samples, key, zero_contrast_fraction=0., add_graded_contrasts=False, num_gradings=4):
    '''
    Generate random binary decision making task inputs and targets. Here, all
    inputs are 3-dimensional, with a constant input channel and two stimulus channels (R / L).

    num_samples             :   number of samples to generate
    key                     :   random number generator key
    zero_contrast_fraction  :   proportion of trials that have no stimulus input (R or L)
    '''
    assert zero_contrast_fraction <= 1.0 and zero_contrast_fraction >= 0.0
    assert num_gradings == 4 or num_gradings == 8

    key, skey, sskey = jr.split(key, 3)

    num_zero_contrast = int(num_samples * zero_contrast_fraction)

    const_channel = jnp.ones((num_samples, 1))

    if not add_graded_contrasts:
        # # Generate a mask for zero-contrast trials
        # mask = jnp.ones(num_samples)
        # masked_indices = jr.permutation(ssskey, jnp.arange(num_samples))[:num_zero_contrast]
        # mask = mask.at[masked_indices].set(0)
        assert num_zero_contrast % 2 == 0
        num_each_stim = num_samples // 2
        num_zero_contrast_per_stim = num_zero_contrast // 2

        # Generate a mask for zero-contrast trials
        mask = jnp.ones(num_each_stim)
        masked_indices = jr.permutation(skey, jnp.arange(num_each_stim))[:num_zero_contrast_per_stim]
        mask = mask.at[masked_indices].set(0)
        mask = jnp.repeat(mask, 2)

        # stim_left = jr.randint(sskey, (num_samples, 1), 0, 2)
        # stim_right = 1 - stim_left
        # inputs = jnp.concatenate([const_channel, stim_right, stim_left], axis=1)
        stim_inputs = jnp.array([[1.0, 0.], [0., 1.0]])
        inputs = jnp.tile(stim_inputs, (num_each_stim, 1))


        permutation = jr.permutation(sskey, inputs.shape[0])
        inputs = inputs[permutation]
        mask = mask[permutation]

        inputs = jnp.concatenate([const_channel, inputs], axis=1)
        big_mask = jnp.concatenate([const_channel, mask.reshape(-1, 1), mask.reshape(-1, 1)], axis=1)

        targets = jnp.argmax(inputs[:, 1:], axis=1)
        targets = targets.reshape(-1, 1)

        inputs *= big_mask


    else:
        # assert num_samples % 4 == 0
        # assert num_zero_contrast % 4 == 0
        assert (num_samples - num_zero_contrast) % num_gradings == 0
        num_each_stim = math.ceil(num_samples / num_gradings)
        num_zero_contrast_per_stim = math.ceil(num_zero_contrast / num_gradings)

        # Generate a mask for zero-contrast trials
        mask = jnp.ones(num_each_stim)
        masked_indices = jr.permutation(skey, jnp.arange(num_each_stim))[:num_zero_contrast_per_stim]
        mask = mask.at[masked_indices].set(0)
        mask = jnp.repeat(mask, num_gradings)

        if num_gradings == 4:
            stim_inputs = jnp.array([[1.0, 0.], [0., 1.0], [0.5, 0.], [0., 0.5]])
        else:
            stim_inputs = jnp.array([[1.0, 0.], [0., 1.0], [0.75, 0], [0, 0.75], [0.5, 0.], [0., 0.5], [0.25, 0], [0, 0.25]])
        inputs = jnp.tile(stim_inputs, (num_each_stim, 1))

        inputs = inputs[:num_samples]
        mask = mask[:num_samples]

        permutation = jr.permutation(sskey, inputs.shape[0])
        inputs = inputs[permutation]
        mask = mask[permutation]
        big_mask = jnp.concatenate([const_channel, mask.reshape(-1, 1), mask.reshape(-1, 1)], axis=1)

        inputs = jnp.concatenate([const_channel, inputs], axis=1)

        targets = jnp.argmax(inputs[:, 1:], axis=1)
        targets = targets.reshape(-1, 1)

        inputs *= big_mask

    return inputs, targets, mask

inputs, outputs, mask = generate_simulated_binary_data(248, jr.PRNGKey(13), zero_contrast_fraction=0.03226, add_graded_contrasts=True, num_gradings=8)


def generate_graded_contrast_inputs(num_samples_per_grading, num_zero_contrast, key, num_grading_levels=10):
    key, skey, sskey = jr.split(key, 3)
    grading_frac = 1./num_grading_levels
    gradings = np.arange(grading_frac, 1 + grading_frac/2, grading_frac)

    stim_inputs = jnp.tile(1.*jnp.eye(2), (num_grading_levels, 1))
    stim_inputs *= jnp.repeat(gradings, 2).reshape(-1, 1)

    stim_inputs = jnp.tile(stim_inputs, (num_samples_per_grading, 1))

    zero_contrast_inputs = jnp.zeros((num_zero_contrast, 2))
    stim_inputs = jnp.concatenate([stim_inputs, zero_contrast_inputs], axis=0)
    mask = jnp.concatenate([jnp.ones(num_samples_per_grading * num_grading_levels * 2), jnp.zeros(num_zero_contrast)])

    targets = jnp.argmax(stim_inputs, axis=1)
    targets = targets.reshape(-1, 1)

    permutation = jr.permutation(skey, stim_inputs.shape[0])
    stim_inputs = stim_inputs[permutation]
    targets = targets[permutation]
    mask = mask[permutation]

    const_channel = jnp.ones((stim_inputs.shape[0], 1))
    inputs = jnp.concatenate([const_channel, stim_inputs], axis=1)

    return inputs, targets, mask


def generate_multisession_graded_data(num_sessions, num_gradings, num_samples_per_grading, num_zero_contrast, key):
    inputs, targets, mask = generate_graded_contrast_inputs(num_samples_per_grading, num_zero_contrast, key, num_grading_levels=num_gradings)
    full_inputs = jnp.tile(inputs, (num_sessions, 1))
    full_targets = jnp.tile(targets, (num_sessions, 1))
    full_mask = jnp.tile(mask, (num_sessions,))
    return full_inputs, full_targets, full_mask


def generate_multisession_binary_data(num_sessions, num_samples_per_session, key, zero_contrast_fraction=0., add_graded_contrasts=False, num_gradings=4):
    '''
    Generate multisession binary decision making task inputs and outputs, consisting
    of the same set of trials being repeated across multiple sessions. Here, all
    inputs are 3-dimensional, with a constant input channel and two stimulus channels (R / L).

    num_sessions            :   number of sessions to generate
    num_samples_per_sess    :   number of samples per session
    key                     :   random number generator key
    zero_contrast_fraction  :   proportion of trials that have no stimulus input (R or L)
    '''
    assert zero_contrast_fraction <= 1.0 and zero_contrast_fraction >= 0.0

    inputs, outputs, mask = generate_simulated_binary_data(num_samples_per_session, key, zero_contrast_fraction=zero_contrast_fraction, add_graded_contrasts=add_graded_contrasts, num_gradings=num_gradings)
    inputs = jnp.tile(inputs, (num_sessions, 1))
    outputs = jnp.tile(outputs, (num_sessions, 1))
    mask = jnp.tile(mask, (num_sessions,))
    return inputs, outputs, mask


def plot_trial_stats(stats, ylabel, benchmark_line=None, color='black', title=None):
    num_models = stats.shape[0]
    plt.figure()
    for i in range(num_models):
        plt.plot(stats[i], color=color, lw=2.3, alpha=0.3)

    plt.plot(jnp.mean(stats, axis=0), color=color, lw=3)
    if benchmark_line is not None:
        plt.axhline(benchmark_line, color='black', lw=2.5, ls='--')
    plt.xlabel('Trial')
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

def plot_R_ZB_L_trial_stats(all_stats, ylabel, benchmark_line=None, title=None, figsize=None):
    R_stats, ZB_stats, L_stats = all_stats
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    for i in range(R_stats.shape[0]):
        plt.plot(R_stats[i], color='green', lw=2.3, alpha=0.15)
    for i in range(ZB_stats.shape[0]):
        plt.plot(ZB_stats[i], color='orange', lw=2.3, alpha=0.15)
    for i in range(L_stats.shape[0]):
        plt.plot(L_stats[i], color='purple', lw=2.3, alpha=0.15)

    plt.plot(jnp.mean(R_stats, axis=0), color='green', lw=3)
    plt.plot(jnp.mean(ZB_stats, axis=0), color='orange', lw=3)
    plt.plot(jnp.mean(L_stats, axis=0), color='purple', lw=3)

    if benchmark_line is not None:
        plt.axhline(benchmark_line, color='black', lw=2.5, ls='--')
    plt.xlabel('Trial')
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)


def calculate_model_agnostic_bias(rightward_statistics, zero_contrast_fraction):
    z = zero_contrast_fraction
    props = rightward_statistics[0] / rightward_statistics[1]
    avg_p_R = (0.5 - z/2.) * props[0] + z * props[1] + (0.5 - z/2.) * props[2]
    return 2 * (avg_p_R - 0.5)      # ensure output ranges from -1 to 1, with 0 being "balanced"

def calculate_model_based_bias(psychometrics):
    middle_idx = psychometrics.shape[1] // 2
    r_slopes = psychometrics[:, -1] - psychometrics[:, middle_idx]    # full right - zero contrast
    l_slopes = psychometrics[:, middle_idx] - psychometrics[:, 0]    # zero contrast - full left

    diffs = r_slopes - l_slopes
    sums = r_slopes + l_slopes
    return np.dot(sums, diffs) / np.sum(sums)

def calculate_overall_acc(rightward_statistics):
    correct_rights = rightward_statistics[0, 0]
    correct_lefts = rightward_statistics[1, 2] - rightward_statistics[0, 2]
    acc = (correct_rights + correct_lefts) / (rightward_statistics[1, 0] + rightward_statistics[1, 2])
    return acc

def calculate_session_accs(rewards, num_samples_per_session, zero_contrast_mask):
    assert len(rewards) % num_samples_per_session == 0
    num_trials = len(rewards)
    num_sessions = num_trials // num_samples_per_session

    zero_contrast_mask = zero_contrast_mask[:num_trials].astype(bool)
    nonmasked_rewards = rewards[zero_contrast_mask]     # access only rewards from non-zero contrast trials
    rewards_by_trial = nonmasked_rewards.reshape(num_sessions, -1)
    accs = jnp.mean(rewards_by_trial, axis=1)
    return accs

def plot_psychometrics(psychometrics, plot_every=1_000):
    plt.figure(figsize=(11, 6))
    colormap = plt.cm.cool #plt.cm.gist_ncar #nipy_spectral, Set1,Paired

    middle_idx = psychometrics.shape[1] // 2
    end_idx = psychometrics.shape[1] - 1

    plt.xticks(ticks=[0, middle_idx/2., middle_idx, (middle_idx + end_idx)/2., end_idx], labels=['Left (100%)', 'Left (50%)', 'Zero', 'Right (50%)', 'Right (100%)'], fontsize=15)
    plt.yticks(fontsize=18)

    n = int(len(psychometrics) / plot_every)
    colors = [colormap(i) for i in np.linspace(0, 1, n)]

    for i in range(n):
        plt.plot(psychometrics[plot_every*i], color=colors[i], label=f'Iteration #{plot_every*i}')

    plt.title('Psychometric Curve', fontsize=20)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

def plot_summary_stats(accs, biases, slope_diffs, stim_DA_biases):
    fig, ax = plt.subplots(4, 1, figsize=(7, 20))

    ax[0].plot(accs)
    ax[0].set_xlabel('Session #', fontsize=18)
    ax[0].set_ylabel('Accuracy', fontsize=18)

    ax[1].plot(biases)
    ax[1].set_xlabel('Iterations', fontsize=18)
    ax[1].set_ylabel('Bias', fontsize=18)

    ax[2].plot(slope_diffs)
    ax[2].set_xlabel('Iterations', fontsize=18)
    ax[2].set_ylabel('Slope Difference (R - L)', fontsize=18)

    ax[3].plot(stim_DA_biases)
    ax[3].set_xlabel('Iterations', fontsize=18)
    ax[3].set_ylabel('Stim DA Bias', fontsize=18)

    for axis in ax.flat:
        axis.tick_params(axis='both', labelsize=18)

def show_histogram(trial_data, bins=10, title=None):
    plt.figure(figsize=(7, 5))
    plt.hist(trial_data, bins=bins)
    if title is not None:
        plt.title(title, fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlabel('Value', fontsize=16)
    plt.show()


def plot_sweep_results(results_dict, threshold=0.1):
    eff_num_trials = results_dict['eff_num_trials']
    print(f'\nEffective number of trials: {eff_num_trials}')
    num_models = len(eff_num_trials)

    slope_diffs = results_dict['slope_diffs']
    biases = np.asarray(results_dict['biases'])
    stim_DA_biases = np.asarray(results_dict['stim_DA_biases'])
    pma_biases = np.asarray(results_dict['partial_model_agnostic_biases'])

    pmb_biases = np.asarray(results_dict['partial_model_based_biases'])

    accuracies = np.asarray(results_dict['accs'])
    slope_diffs = np.asarray(slope_diffs)

    idx_to_color={0 : 'darkgreen', 1: 'darkorange', 2: 'darkviolet'}

    fig, axs = plt.subplots(4, 1, figsize=(8, 16))
    plt.xlabel('Iterations', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    axs[0].set_ylabel('Accuracy', fontsize=18)
    axs[1].set_ylabel('Slope Difference (R - L)', fontsize=18)
    axs[2].set_ylabel('Bias', fontsize=18)
    axs[3].set_ylabel('Stim DA Bias', fontsize=18)

    axs[0].set_xlim(0, 40)
    axs[1].set_xlim(0, 10_000)
    axs[2].set_xlim(0, 10_000)
    axs[3].set_xlim(0, 10_000)

    avg_accs = np.zeros((3, accuracies.shape[-1]))
    avg_slope_diffs = np.zeros((3, slope_diffs.shape[-1]))
    avg_biases = np.zeros_like(avg_slope_diffs)
    avg_stim_DA_biases = np.zeros_like(avg_slope_diffs)
    bias_stats = np.zeros((3, 1))

    for i in range(len(pma_biases)):
        trials = eff_num_trials[i]
        pma_bias = pma_biases[i]
        idx = None
        color = None
        if pma_bias >= threshold:
            idx = 0
            color = idx_to_color[0]
            bias_stats[0] += 1
        elif pma_bias <= -threshold:
            idx = 2
            color = idx_to_color[2]
            bias_stats[2] += 1
        else:
            idx = 1
            color = idx_to_color[1]
            bias_stats[1] += 1

        curr_accs = accuracies[i, :int(trials/250)]
        axs[0].plot(curr_accs, color=color, alpha=0.5)
        axs[0].plot(int(trials/250)-1, curr_accs[-1], marker='o', color='cadetblue', alpha=0.5)

        axs[1].plot(slope_diffs[i, :trials], color=color, alpha=0.5)
        axs[1].plot(trials-1, slope_diffs[i, trials-1], marker='o', color='cadetblue', alpha=0.5)

        axs[2].plot(biases[i, :trials], color=color, alpha=0.5)
        axs[2].plot(trials-1, biases[i, trials-1], marker='o', color='cadetblue', alpha=0.5)

        axs[3].plot(stim_DA_biases[i, :trials], color=color, alpha=0.5)
        axs[3].plot(trials-1, stim_DA_biases[i, trials-1], marker='o', color='cadetblue', alpha=0.5)

        avg_accs[idx] += accuracies[i]
        avg_slope_diffs[idx] += slope_diffs[i]
        avg_biases[idx] += biases[i]
        avg_stim_DA_biases[idx] += stim_DA_biases[i]

    avg_accs /= bias_stats
    avg_slope_diffs /= bias_stats
    avg_biases /= bias_stats
    avg_stim_DA_biases /= bias_stats

    for i in range(3):
        axs[0].plot(avg_accs[i], color=idx_to_color[i], linewidth=3)
        axs[1].plot(avg_slope_diffs[i], color=idx_to_color[i], linewidth=3)
        axs[2].plot(avg_biases[i], color=idx_to_color[i], linewidth=3)
        axs[3].plot(avg_stim_DA_biases[i], color=idx_to_color[i], linewidth=3)

    plt.title('Model Sweep Results')
    plt.show()
    print(bias_stats)


def flatten_params(w):
    params_matrix = None
    for key in w:
        if params_matrix is None:
            params_matrix = w[key].reshape(w[key].shape[0], -1)
        else:
            params_matrix = jnp.concatenate((params_matrix, w[key].reshape(w[key].shape[0], -1)), axis=-1)
    return params_matrix 


def turn_pytree_to_arr(tree):
    arr = jnp.concatenate([tree[key].reshape(-1) for key in tree.keys()])
    return arr

def combine_batch_weights(ws):
    B = ws['W2'].shape[0]
    T = ws['W2'].shape[1]
    ws_arr = jnp.concatenate((ws['W1_diag'], ws['W2'].reshape(B, T, -1)), axis=-1)
    return ws_arr 


def realign_weight_trajectories(true_ws, inferred_ws):
    '''
    Arguments:
    true_ws     :       true weight trajectories (num_timesteps x num_params)
    inferred_ws :       inferred weight trajectories (num_timesteps x num_params)
    '''
    sign_true_ws = jnp.sign(true_ws)
    true_signs = jnp.sign(jnp.mean(jnp.mean(sign_true_ws, axis=0).reshape(3, 3), axis=0))    # decide whether trajectory is 'positive' or 'negative'
    true_signs = jnp.tile(true_signs, 3)

    sign_inferred_ws = jnp.sign(inferred_ws)
    inferred_signs = jnp.sign(jnp.mean(jnp.mean(sign_inferred_ws, axis=0).reshape(3, 3), axis=0))
    inferred_signs = jnp.tile(inferred_signs, 3)

    sign_flip_map = true_signs / inferred_signs

    inferred_ws_realigned = inferred_ws * sign_flip_map.reshape(1, -1)

    return inferred_ws_realigned


def compute_AIC(nll, num_params):
    return 2 * nll + 2 * num_params 
