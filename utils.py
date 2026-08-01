import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
from jax.tree_util import tree_map
import optax
import numpy as np

import os
import pickle as pkl
from scipy import stats
from scipy.signal import argrelmax

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

import functools
import random
import math
from scipy.optimize import curve_fit
from globals import NUM_GRADING_LEVELS


def flip_outputs(y):
    assert y.ndim == 1
    flipped = 1 - y
    return flipped.astype(int)


def tanh_p(x, p=5.):
    return jnp.tanh(p * x) / jnp.tanh(p)


def check_writable(directory):
    """Fail fast if `directory` is not writable.

    Called before fitting starts: a bad path or an exhausted disk quota should cost a
    second, not a multi-hour job that dies at the final `pkl.dump`.
    """
    probe = os.path.join(directory, f'.writetest_{os.getpid()}')
    try:
        with open(probe, 'wb') as f:
            f.write(b'ok')
        os.remove(probe)
    except OSError as e:
        raise SystemExit(f'ERROR: cannot write results to {directory}: {e}')


def to_numpy(obj):
    """Recursively convert JAX arrays to NumPy inside containers.

    Saved records are then plain NumPy/Python objects, so they load anywhere without
    JAX installed and do not depend on JAX's pickle format staying stable.
    """
    if isinstance(obj, jnp.ndarray):
        return np.asarray(obj)
    if isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(v) for v in obj)
    return obj


def pad_sequence(arr, target_len):
    pad_width = [(0, target_len - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
    return jnp.pad(arr, pad_width)


def pad_trial_data(sequences):
    lengths = jnp.array([s[0].shape[0] for s in sequences])
    T_max = int(lengths.max())

    inputs  = jnp.stack([pad_sequence(s[0], T_max) for s in sequences])  # (B, T_max, D)
    outputs = jnp.stack([pad_sequence(s[1], T_max) for s in sequences])  # (B, T_max)
    rewards = jnp.stack([pad_sequence(s[2], T_max) for s in sequences])  # (B, T_max)

    return inputs, outputs, rewards, lengths 


def make_train_mask(num_trials, block_length=20, test_fraction=0.1):
    # sample blocks of test trials randomly in blocks of roughly length 20, with 10% of trials as test
    # returns a boolean mask of shape (num_trials,) where True indicates a training trial and False indicates a test trial
    num_blocks = math.ceil(num_trials / block_length)
    block_indices = list(range(num_blocks))
    num_test_blocks = math.ceil(test_fraction * num_blocks)
    test_block_indices = random.sample(block_indices, num_test_blocks)
    mask = np.ones(num_trials, dtype=bool)
    for block_idx in test_block_indices:
        start = block_idx * block_length
        end = min(start + block_length, num_trials)
        mask[start:end] = False
    return mask


def make_kfold_train_masks(num_trials, key, K=5, block_length=20):
    '''
    Partition the trials into K disjoint, exhaustive cross-validation folds made up of
    contiguous blocks of `block_length` trials, and return one training mask per fold.

    Blocks are randomly assigned to folds (via `key`), so each fold's held-out trials are
    scattered across the whole learning trajectory rather than concentrated in one epoch.
    Fold sizes differ by at most one block, and every trial is held out in exactly one fold,
    so summing the K held-out NLLs and dividing by `num_trials` gives the cross-validated
    NLL per trial.

    Arguments:
    num_trials   : total number of trials for this animal
    key          : JAX PRNG key controlling the (reproducible) block-to-fold assignment.
                   Pass the same key across model classes so all models are scored on
                   identical folds, making model comparisons paired.
    K            : number of folds (each holds out ~1/K of the trials)
    block_length : number of contiguous trials per block

    Returns:
    a list of K boolean arrays of shape (num_trials,), where entry k is the training mask
    for fold k: True indicates a training trial and False indicates a held-out test trial
    (same convention as `make_train_mask`)
    '''
    num_blocks = math.ceil(num_trials / block_length)
    assert num_blocks >= K, (
        f'Need at least K={K} blocks to form K non-empty folds, but num_trials={num_trials} '
        f'with block_length={block_length} gives only {num_blocks} blocks. Folds without any '
        f'held-out trials would silently contribute a test NLL of 0.'
    )

    # randomly assign blocks to folds; array_split gives K groups whose sizes differ by <= 1 block
    block_order = jr.permutation(key, num_blocks)
    fold_blocks = jnp.array_split(block_order, K)

    # map each block to its fold, then each trial to its block's fold
    block_to_fold = jnp.zeros(num_blocks, dtype=jnp.int32)
    for k, blocks in enumerate(fold_blocks):
        block_to_fold = block_to_fold.at[blocks].set(k)

    trial_folds = block_to_fold[jnp.arange(num_trials) // block_length]

    # fold k trains on every trial NOT assigned to fold k
    return [trial_folds != k for k in range(K)]

# xs = make_kfold_train_masks(13678, key=jr.PRNGKey(1), K=5, block_length=20)
# for x in xs:
#     # print(x.astype(int))
#     print(x.sum() / len(x))

# mask = make_train_mask(13_762, block_length=20, test_fraction=0.1)
# num_trains = np.sum(mask)
# num_tests = np.sum(~mask)
# print(num_tests / (num_trains + num_tests))  # should be close to 0.1

# keys = jr.split(jr.PRNGKey(0), 10)
# sequences = []
# Ts = [2, 4, 5]
# for T in Ts:
#     seq = (jr.normal(keys[0], (T, 3)), jr.randint(keys[1], (T,), 0, 2), jr.randint(keys[2], (T,), 0, 2))
#     print(seq)
#     sequences.append(seq)

# lengths = jnp.array([s[0].shape[0] for s in sequences])
# print(lengths)
# T_max = int(lengths.max())

# def pad(arr, target_len):
#     pad_width = [(0, target_len - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
#     return jnp.pad(arr, pad_width)

# inputs  = jnp.stack([pad(s[0], T_max) for s in sequences])  # (B, T_max, D)
# outputs = jnp.stack([pad(s[1], T_max) for s in sequences])  # (B, T_max, ...)
# rewards = jnp.stack([pad(s[2], T_max) for s in sequences])  # (B, T_max, ...)

# print(inputs)
# print(outputs)
# print(rewards)
# print(1./0)


def make_legend(models_and_mouse=False, mark_expert=False):
    if models_and_mouse:
        fig, ax = plt.subplots(figsize=(4.5, 0.5))
        ax.axis('off')

        handles = [
            mlines.Line2D([], [], color="#ea2626", linewidth=2, linestyle='-',  label='True Mouse'),
            mlines.Line2D([], [], color='#2166AC', linewidth=2, linestyle='--', label='1-layer Linear'),
            mlines.Line2D([], [], color='#F4A736', linewidth=2, linestyle='--', label='2-layer Linear'),
            mlines.Line2D([], [], color='#2CA02C', linewidth=2, linestyle='--', label='2-layer Nonlinear'),
        ]

        legend = ax.legend(
            handles=handles,
            loc='center',
            ncol=2,
            frameon=False,
            fontsize=11,
            handlelength=2.5,
            columnspacing=1.5,
            handletextpad=0.5,
        )

        plt.tight_layout()
        plt.savefig('models_and_mouse_legend.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        MODEL_COLORS = ['#2166AC', '#F4A736', '#2CA02C']
        MODEL_NAMES  = ['1-Layer Linear', '2-Layer Linear', '2-Layer Nonlinear']

        legend_handles = [mpatches.Patch(color=c, label=n)
                        for c, n in zip(MODEL_COLORS, MODEL_NAMES)]
        ncols = 3
        if mark_expert:
            legend_handles += [Line2D([0], [0], marker='o', color='none', ms=8,
                                markeredgecolor='black', markeredgewidth=1.2, label='Expert')]
            ncols = 2

        fig, ax = plt.subplots(figsize=(4, 1))
        ax.axis('off')
        ax.legend(handles=legend_handles, fontsize=23, frameon=False,
                loc='center', ncol=ncols, handlelength=2., handleheight=1.2,
                columnspacing=1.2)

        plt.tight_layout()
        plt.savefig(f'models_legend_wexperts{mark_expert}.pdf', dpi=300, bbox_inches='tight')
        plt.show()

# make_legend(mark_expert=True)


def find_model_success_rate(aic_data, model_idx):
    best_idxs = jnp.argmin(aic_data, axis=1)
    hits = (best_idxs == model_idx)
    return 1.*jnp.sum(hits)/len(aic_data)


def find_winners(aic_data, idx1, idx2):
    winners = []
    for i in range(len(aic_data)):
        if aic_data[i, idx1] <= aic_data[i, idx2]:
            winners.append(idx1)
        else:
            winners.append(idx2)
    
    return winners


def get_nonlinearity(nln_name):
    nonlinearities = {
        'erf': jax.lax.erf,
        'linear': lambda x: x,
        'tanh': jnp.tanh,
        'relu': jax.nn.relu,
    }
    try:
        return nonlinearities[nln_name]
    except KeyError:
        raise ValueError(f"Unknown nonlinearity: {nln_name}")
    

def optimal_sign_flip(y, z):
    """
    Find the sign flip vector s in {-1, +1}^3 (over the last axis/columns)
    that minimizes ||y - z * s||^2, summed over all other dimensions.

    Args:
        y, z: jnp arrays of shape (T, 3, 3) (two-layer net)

    Returns:
        s:        sign flip vector of shape (3,), values in {-1, +1}
        sq_error: scalar, minimized total squared error using optimal s
    """
    # Error per column for s=+1 vs s=-1, summed over T and rows
    # Shape: (3,)
    err_pos = jnp.sum((y - z) ** 2, axis=(0, 1))   # s_j = +1
    err_neg = jnp.sum((y + z) ** 2, axis=(0, 1))   # s_j = -1

    s = jnp.where(err_pos <= err_neg, 1.0, -1.0)   # shape (3,)
    sq_error = jnp.sum(jnp.minimum(err_pos, err_neg))

    return s, sq_error
    

def compute_variances_along_trajectory(trajectories):
    return jnp.var(trajectories, axis=0)


def get_pearson_correlations(X, y, ignore_nans=True):
    '''
    Arguments:
    X : n x T
    y : T
    '''
    if ignore_nans:
        valid = ~np.isnan(y)
        y = y[valid]
        X = X[:, valid]

    # Demean
    X_centered = X - X.mean(axis=1, keepdims=True)
    y_centered = y - y.mean()

    # Numerator: dot product of each row with y
    numerator = X_centered @ y_centered  # shape (n,)

    # Denominator: product of standard deviations
    X_std = np.sqrt((X_centered ** 2).sum(axis=1))  # shape (n,)
    y_std = np.sqrt((y_centered ** 2).sum())         # scalar

    return numerator / (X_std * y_std)  # shape (n,)
    

def compute_coeff_of_determination(x_model, x_true):
    ss_res = np.nansum((x_true - x_model) ** 2)
    ss_tot = np.nansum((x_true - np.nanmean(x_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def compute_r2_distribution(model_trajectories, true_trajectory):
    r2s = []
    for i in range(model_trajectories.shape[0]):
        r2 = compute_coeff_of_determination(model_trajectories[i], true_trajectory)
        r2s.append(r2)
    return np.asarray(r2s)


def visualize_r2_distributions(r2s_arrays, plot_colors, filepath=None,
                               xlabel=r'$R^2$ (Coeff. of Determination)', xticks=None, xmin=None):

    # create density plot of r2s
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    mins = [0.]
    # r2_mins = [np.min(r2s) for r2s in r2s_arrays]
    # min_r2 = np.min(r2_mins)

    for r2s, plot_color in zip(r2s_arrays, plot_colors):
        kde = stats.gaussian_kde(r2s, bw_method='scott')
        x = np.linspace(r2s.min() - 0.1, 1.0, 300)
        y = kde(x)
        ax.plot(x, y, color=plot_color, lw=1.8, zorder=3)
        ax.fill_between(x, y, color=plot_color, alpha=0.2, linewidth=0, zorder=2)

        # ax.axvline(r2s.mean(), color=plot_color, lw=1.2, linestyle='--', zorder=4,
        #         label=f'Mean $R^2 = {r2s.mean():.2f}$')
        ax.axvline(0, color='gray', lw=0.8, linestyle=':', zorder=1)

        # find largest R^2 peak in density
        peak_idxs = argrelmax(y, order=10)[0]
        if len(peak_idxs) == 0:
            peak_idxs = [np.argmax(y)]  # fallback if no local max found
        peak_x = x[peak_idxs[np.argmax(peak_idxs)]]  # rightmost peak
        mins.append(peak_x)

        ax.axvline(peak_x, color=plot_color, lw=1.0, linestyle='--', zorder=4, alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=23)
    ax.set_ylabel('Density', fontsize=24)
    ax.set_xlim(np.min(np.asarray(mins))-1., 1.0)
    if xmin is not None:
        ax.set_xlim(xmin, 1.0)
    ax.set_ylim(bottom=0)

    if xticks is not None:
        ax.set_xticks(xticks)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.tick_params(labelsize=21.5, length=3, width=0.8)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False)

    plt.tight_layout()
    if filepath is not None:
        fig.savefig(f'{filepath}', dpi=300, bbox_inches='tight')
    else:
        plt.show()



def run_linear_regression(xs, ys):
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    r_squared = r_value**2
    x_line = np.array([xs.min(), xs.max()])
    y_line = slope * x_line + intercept
    return x_line, y_line, r_squared

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def get_sigmoid_fit(x, y):
    popt, pcov = curve_fit(sigmoid, x, y, p0=[y.max(), np.median(x), 1, y.min()])
    # Extract parameters
    L, x0, k, b = popt
    # Generate fitted data
    y_fit = sigmoid(x, L, x0, k, b)
    return y_fit, popt

def calculate_accs_per_session(rewards, trials_per_session):
    rewards = np.asarray(rewards)
    trials_per_session = np.asarray(trials_per_session)

    num_sessions = len(trials_per_session)
    accs = []
    start_idx = 0
    for i in range(num_sessions):
        end_idx = start_idx + trials_per_session[i]
        session_rewards = rewards[start_idx:end_idx]
        acc = np.mean(session_rewards)
        accs.append(acc)
        start_idx = end_idx
    return np.array(accs)


# def calculate_running_avg_quantity(all_measurements, window):
#     ndim = all_measurements.ndim 
#     if ndim == 1:
#         all_measurements = all_measurements.reshape(1, -1)
#     running_sums = np.cumsum(all_measurements, axis=1)
    
#     # Pad with zeros on the left for convenience
#     running_sums = np.pad(running_sums, ((0,0),(1,0)), mode='constant')
    
#     # Window sums
#     window_sums = running_sums[:, window:] - running_sums[:, :-window]
    
#     running_accs = window_sums / window

#     # for convenience, pad first 'window - 1' trials with NaNs
#     pad = np.full((all_measurements.shape[0], window-1), np.nan)
#     running_accs = np.concatenate([pad, running_accs], axis=1)

#     if ndim == 1:
#         running_accs = running_accs.reshape(-1)
    
#     return running_accs 

def get_zero_contrast_mask(inputs):
    zero_contrast_mask = (inputs[:, 1] == 0) & (inputs[:, 2] == 0)
    return 1-zero_contrast_mask


def calculate_running_avg_quantity(all_measurements, window, mask=None):
    ndim = all_measurements.ndim
    if ndim == 1:
        all_measurements = all_measurements.reshape(1, -1)

    if mask is not None:
        valid_mask = mask.astype(float)  # shape (n_trials,), broadcasts over rows
    else:
        valid_mask = np.ones(all_measurements.shape[1])
    valid_mask = valid_mask.reshape(1, -1)  # ensure shape is (1, n_trials) for broadcasting

    filled = all_measurements * valid_mask

    # Cumulative sums of values and valid counts
    cumsum_vals  = np.cumsum(filled, axis=1)
    cumsum_valid = np.cumsum(valid_mask, axis=1)

    # Pad left with zeros for the sliding-window subtraction trick
    cumsum_vals  = np.pad(cumsum_vals, ((0, 0), (1, 0)), mode='constant')
    cumsum_valid = np.pad(cumsum_valid, ((0, 0), (1, 0)), mode='constant')

    # Window sums and counts
    window_sums   = cumsum_vals[:,  window:] - cumsum_vals[:,  :-window]
    window_counts = cumsum_valid[:, window:] - cumsum_valid[:, :-window]

    # Divide; windows with zero valid trials → NaN
    with np.errstate(invalid='ignore'):
        running_accs = np.where(window_counts > 0, window_sums / window_counts, np.nan)

    # Pad first (window - 1) positions with NaN, consistent with original behaviour
    pad = np.full((all_measurements.shape[0], window - 1), np.nan)
    running_accs = np.concatenate([pad, running_accs], axis=1)

    if ndim == 1:
        running_accs = running_accs.squeeze(0)

    return running_accs


def calculate_batched_quantity_per_session(all_measurements, trials_per_session, mask=None):
    '''
    Arguments:
    all_measurements: a 1D or 2D array of shape (num_models, num_trials) containing the given quantity's measurement for each model across all trials
    trials_per_session: a 1D array of length num_sessions containing the number of trials in each session
    '''
    all_measurements = np.asarray(all_measurements)
    trials_per_session = np.asarray(trials_per_session)

    ndim = all_measurements.ndim
    if ndim == 1:
        all_measurements = all_measurements.reshape(1, -1)

    if mask is not None:
        valid_mask = mask.astype(float)  # shape (n_trials,), broadcasts over rows
    else:
        valid_mask = np.ones(all_measurements.shape[1])

    num_sessions = len(trials_per_session)
    batched_measurements = []
    start_idx = 0
    for i in range(num_sessions):
        curr_mask = valid_mask[start_idx:start_idx + trials_per_session[i]]
        end_idx = start_idx + trials_per_session[i]
        session_measurements = all_measurements[:, start_idx:end_idx]
        session_mean = np.sum(session_measurements * curr_mask, axis=1) / np.sum(curr_mask)
        batched_measurements.append(session_mean)
        start_idx = end_idx

    session_avgs = np.array(batched_measurements).T  # shape (num_models, num_sessions)

    if ndim == 1:
        session_avgs = session_avgs.squeeze(0)

    return session_avgs


def calculate_avg_acc_diff_per_trial(session_accs, trials_per_session, baseline_acc=0.5):
    acc_diffs = np.diff(session_accs)
    cumulative_acc_diffs = np.sum(acc_diffs / trials_per_session[1:]) + (session_accs[0] - baseline_acc) / trials_per_session[0]
    avg_acc_diff_per_trial = cumulative_acc_diffs / len(session_accs)
    return avg_acc_diff_per_trial

def calculate_avg_acc_per_trial(session_accs, trials_per_session):
    return np.mean(session_accs / trials_per_session)

def calculate_trial_weighted_acc_diff(session_accs, trials_per_session, baseline_acc=0.5):
    acc_diffs = np.diff(session_accs)
    cumulative_acc_diffs = np.sum(acc_diffs * trials_per_session[1:]) + (session_accs[0] - baseline_acc) * trials_per_session[0]
    avg_trial_weighted_acc = cumulative_acc_diffs / np.sum(trials_per_session)
    return avg_trial_weighted_acc

def calculate_trial_averaged_acc(rewards, inputs, baseline_acc=0.5):
    mask = get_zero_contrast_mask(inputs)
    return 2 * (jnp.sum(mask * rewards) / jnp.sum(mask) - baseline_acc)

def calculate_normalized_avg_acc(session_accs, baseline_acc=0.5):
    avg_acc = np.mean(session_accs)
    return 2 * (avg_acc - baseline_acc)


def measure_sigmoidal_learning_speed(session_accs):
    '''
    Fit a sigmoid to the session accuracies and return the slope parameter.
    '''
    x_data = np.arange(len(session_accs)) + 1
    y_data = session_accs

    try:
        popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=[y_data.max(), np.median(x_data), 0.1, y_data.min()], bounds=([0., -1000., 0., 0.], [1., 1000., 1000., 1.])) 
        L, x0, k, b = popt
        learning_ceiling = L + b    # final accuracy in large time limit
        gain = k                    # gain param of exponential
        speed = k * L / 4           # max slope of sigmoid
        y_fit = sigmoid(x_data, L, x0, k, b)
    except RuntimeError:
        y_fit = None 
        learning_ceiling, speed = None, None
        popt = None


    return speed, learning_ceiling, y_fit, popt


def extract_specific_grading_levels(inputs):
    # include zero contrast too
    right_contrasts = inputs[:, 1]
    left_contrasts = inputs[:, 2]
    contrasts = np.concatenate([right_contrasts, left_contrasts])
    unique_contrasts = np.unique(contrasts)
    unique_contrasts = np.sort(unique_contrasts)
    return unique_contrasts

def convert_inds_to_grading_levels(inds, num_grading_levels=NUM_GRADING_LEVELS):
    grading_unit = 1.0 / num_grading_levels
    zero_idx = num_grading_levels
    grading_levels = (inds - zero_idx) * grading_unit
    return grading_levels


# generated with help from Claude, and verified against older implementation
def vectorized_generate_per_session_psychometrics(inputs, outputs, trials_per_session, num_grading_levels,
                                                  specific_grading_levels=[0.0625, 0.125, 0.25, 1.0]):
    """
    Generate per-session psychometric curves at specified contrast levels.

    Parameters
    ----------
    inputs                  : (num_trials, input_dim) array; col 0 is non-stimulus,
                              cols 1+ are [right_contrast, left_contrast]
    outputs                 : (num_trials,) or (num_trials, 1); 0 = rightward choice
    trials_per_session      : (num_sessions,) trial counts per session
    num_grading_levels      : number of contrast levels excluding zero
    specific_grading_levels : contrast values to track; None returns all levels

    Returns
    -------
    psychometrics   : (num_sessions, num_levels) p(right) per session and level
    col_inds        : column indices into the full contrast array that were selected
    rights          : (num_sessions, num_levels) rightward choice counts
    totals          : (num_sessions, num_levels) total trial counts
    """

    if specific_grading_levels is None:
        specific_grading_levels = extract_specific_grading_levels(inputs)

    grading_unit = 1.0 / num_grading_levels
    zero_idx     = num_grading_levels
    num_bins     = 2 * num_grading_levels + 1
    num_sessions = len(trials_per_session)

    # --- Vectorise over all trials at once -----------------------------------
    # signed_contrast > 0 → right stimulus, < 0 → left stimulus, 0 → zero contrast
    signed_contrast = inputs[:, 1] - inputs[:, 2]               # (num_trials,)
    bin_indices     = (zero_idx + np.rint(signed_contrast / grading_unit)).astype(int)
    went_right      = (outputs.ravel() == 0).astype(float)      # 1 where rightward

    # --- Accumulate per session using bincount (fast, no Python loop) ---------
    endpoints = np.concatenate(([0], np.cumsum(trials_per_session)))
    rights = np.zeros((num_sessions, num_bins))
    totals = np.zeros((num_sessions, num_bins))

    for i, (lo, hi) in enumerate(zip(endpoints[:-1], endpoints[1:])):
        sess_bins = bin_indices[lo:hi]
        totals[i] = np.bincount(sess_bins, minlength=num_bins)
        rights[i] = np.bincount(sess_bins, weights=went_right[lo:hi], minlength=num_bins)

    # --- Select contrast columns ---------------------------------------------
    intensities = np.rint(np.asarray(specific_grading_levels) / grading_unit).astype(int)
    nonzero = intensities[intensities > 0]

    parts = [zero_idx - nonzero[::-1]]
    if 0 in intensities:
        parts.append([zero_idx])
    parts.append(zero_idx + nonzero)

    col_inds = np.concatenate(parts)

    rights = rights[:, col_inds]
    totals = totals[:, col_inds]

    with np.errstate(invalid='ignore', divide='ignore'):
        psychometrics = np.where(totals > 0, rights / totals, np.nan)

    return psychometrics, col_inds, rights, totals


def generate_per_session_psychometrics(inputs, outputs, trials_per_session, num_grading_levels, 
                                       specific_grading_levels=[0.0625, 0.125, 0.25, 1.0]):
    '''
    Generate psychometric curves for each session.

    inputs              :   full set of inputs (num_trials x input_dim)
    outputs             :   full set of outputs (num_trials x 1)
    trials_per_session  :   an array containing the number of trials in each session
    num_grading_levels  :   number of graded contrast levels (excluding zero-contrast)
    '''
    num_sessions = len(trials_per_session)
    grading_unit = 1. / num_grading_levels

    rights_by_grading = np.zeros((num_sessions, 2 * num_grading_levels + 1))
    totals_by_grading = np.zeros((num_sessions, 2 * num_grading_levels + 1))
    zero_grading_idx = num_grading_levels

    session_endpoints = np.cumsum(trials_per_session)
    session_endpoints = np.concatenate((np.array([0]), session_endpoints))      # an array of all the session endpoints
    
    for i in range(num_sessions):
        lower = session_endpoints[i]
        upper = session_endpoints[i+1]
        curr_inputs = inputs[lower:upper]
        curr_outputs = outputs[lower:upper]
        for j in range(len(curr_inputs)):
            inp = curr_inputs[j]
            stim_inp = inp[1:]
            out = curr_outputs[j]

            grading_intensity = int(np.sum(stim_inp) / grading_unit)

            if (stim_inp[0] != 0):
                if out == 0:
                    rights_by_grading[i, zero_grading_idx + grading_intensity] += 1.
                totals_by_grading[i, zero_grading_idx + grading_intensity] += 1.
            elif (stim_inp[1] != 0):
                if out == 0:
                    rights_by_grading[i, zero_grading_idx - grading_intensity] += 1.
                totals_by_grading[i, zero_grading_idx - grading_intensity] += 1
            else:
                # zero contrast trial
                if out == 0:
                    rights_by_grading[i, zero_grading_idx] += 1.
                totals_by_grading[i, zero_grading_idx] += 1.
    
    # psychometrics = rights_by_grading / totals_by_grading
    with np.errstate(invalid='ignore', divide='ignore'):
        psychometrics = np.where(totals_by_grading > 0,
                                rights_by_grading / totals_by_grading,
                                np.nan)
    inds = np.arange(len(psychometrics))

    if specific_grading_levels is not None:
        grading_intensities = np.asarray(specific_grading_levels) / grading_unit 
        grading_intensities = grading_intensities.astype(int)

        left_inds = zero_grading_idx - np.flip(grading_intensities)
        right_inds = zero_grading_idx + grading_intensities
        inds = np.concatenate((left_inds, right_inds))

        psychometrics = psychometrics[:, inds]
        rights_by_grading = rights_by_grading[:, inds]
        totals_by_grading = totals_by_grading[:, inds]

    return psychometrics, inds, rights_by_grading, totals_by_grading


def calculate_simple_mouse_ma_bias(choices):
    '''
    choices : binary outputs (0 = right, 1 = left); shape is either T, or (B, T) if over batch
    '''

    num_rights = np.sum(choices == 0, axis=-1)
    num_trials = choices.shape[-1]
    return (num_rights / num_trials - 0.5) * 2


def get_biases_per_session(outputs, trials_per_session, calculate_within_session=False):
    trial_endpts = np.cumsum(trials_per_session) 
    trial_startpts = np.concatenate((np.array([0]), np.cumsum(trials_per_session)))
    num_sessions = len(trials_per_session)
    biases = np.zeros(num_sessions)
    for i in range(num_sessions):
        # curr_inputs = None 
        curr_outputs = None 
        if calculate_within_session:
            # curr_inputs = inputs[trial_startpts[i]:trial_startpts[i+1]]
            curr_outputs = outputs[trial_startpts[i]:trial_startpts[i+1]]
        else:
            # curr_inputs = inputs[:trial_endpts[i]]
            curr_outputs = outputs[:trial_endpts[i]]
        
        # curr_bias = compute_model_agnostic_bias_with_inputs_and_outputs(curr_inputs, curr_outputs, zero_contrast_fraction)
        curr_bias = calculate_simple_mouse_ma_bias(curr_outputs)
        biases[i] = curr_bias 
    return biases 


def get_biases_per_session_batched(all_outputs, trials_per_session, calculate_within_session=False):
    n = all_outputs.shape[0]
    trial_endpts = np.concatenate((np.array([0]), np.cumsum(trials_per_session)))
    num_sessions = len(trials_per_session)
    biases = np.zeros((n, num_sessions))

    for i in range(num_sessions):
        curr_outputs = None 
        if calculate_within_session:
            curr_outputs = all_outputs[:, trial_endpts[i]:trial_endpts[i+1]]
        else:
            curr_outputs = all_outputs[:, :trial_endpts[i+1]]
        
        curr_bias = calculate_simple_mouse_ma_bias(curr_outputs)
        biases[:, i] = curr_bias 
    return biases 


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


def generate_randomized_multisession_graded_data(num_sessions, num_gradings, num_samples_per_grading, 
                                                 num_zero_contrast_per_session, key):
    full_inputs, full_targets, full_mask = None, None, None
    for i in range(num_sessions):
        _, key = jr.split(key, 2)
        inputs, targets, mask = generate_graded_contrast_inputs(num_samples_per_grading, num_zero_contrast_per_session, key, num_gradings)

        if full_inputs is None:
            full_inputs = inputs
            full_targets = targets 
            full_mask = mask 
        else:
            full_inputs = jnp.concatenate([full_inputs, inputs], axis=0)
            full_targets = jnp.concatenate([full_targets, targets], axis=0)
            full_mask = jnp.concatenate([full_mask, mask])

    return full_inputs, full_targets, full_mask

generate_batched_randomized_multisession_graded_data = jax.vmap(generate_randomized_multisession_graded_data, in_axes=(None, None, None, None, 0))


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

    inputs, targets, mask = generate_simulated_binary_data(num_samples_per_session, key, zero_contrast_fraction=zero_contrast_fraction, add_graded_contrasts=add_graded_contrasts, num_gradings=num_gradings)
    inputs = jnp.tile(inputs, (num_sessions, 1))
    targets = jnp.tile(targets, (num_sessions, 1))
    mask = jnp.tile(mask, (num_sessions,))
    return inputs, targets, mask


def generate_full_input_set(num_grading_levels):
    grading = 1./num_grading_levels
    gradings = jnp.arange(0, 1. + grading, grading)
    R_column = jnp.concatenate([jnp.zeros(num_grading_levels), gradings])
    L_column = jnp.flip(R_column)
    const_column = jnp.ones(2 * num_grading_levels + 1)
    xs = jnp.array([const_column, R_column, L_column]).T
    xs = xs.astype(jnp.float32)
    return xs


def compute_rightward_statistics(inputs, choices):
    rightward_statistics = np.zeros((2, 3))
    ntrials = inputs.shape[0]
    for i in range(ntrials):
        input = inputs[i]
        choice = choices[i]     # should be 0 if input rightward, and 1 if input leftward
        assert (input[1] == 0) or (input[2] == 0)

        if input[1] > 0:
            rightward_statistics[1, 0] += 1
            rightward_statistics[0, 0] += (1 - choice)
        elif input[2] > 0:
            rightward_statistics[1, 2] += 1
            rightward_statistics[0, 2] += (1 - choice)
        else:
            rightward_statistics[1, 1] += 1
            rightward_statistics[0, 1] += (1 - choice)
    return rightward_statistics


def compute_model_agnostic_bias_with_inputs_and_outputs(inputs, outputs, zero_contrast_fraction):
    outputs = outputs.astype(jnp.int32)
    rightward_stats = compute_rightward_statistics(inputs, outputs)
    ma_bias = calculate_model_agnostic_bias(rightward_stats, zero_contrast_fraction=zero_contrast_fraction)
    return ma_bias


def calculate_model_agnostic_bias(rightward_statistics, zero_contrast_fraction):
    tot_num_trials = rightward_statistics[1, :].sum()
    R_prop = rightward_statistics[1, 0] / tot_num_trials
    L_prop = rightward_statistics[1, 2] / tot_num_trials
    z = zero_contrast_fraction
    props = rightward_statistics[0] / np.maximum(rightward_statistics[1], 1.)   # avoid edge cases where denominator is 0
    avg_p_R = R_prop * props[0] + z * props[1] + L_prop * props[2]
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

def plot_psychometrics(psychometrics, plot_every=1_000, use_colormap=True, filepath=None, 
                       title=None, label_word='Iteration', use_witten_scheme=False, use_ibl_ticks=False,
                       custom_inds=None, custom_gradings=None):
    assert label_word in ['Iteration', 'Session', 'Trial']
    fig, ax = plt.subplots(figsize=(6, 4))
    colormap = plt.cm.plasma
    inds = None 

    if not use_witten_scheme:
        if use_ibl_ticks:
            ibl_inds = [0, 12, 14, 15, 17, 18, 20, 32]
            ax.set_xticks(ibl_inds)
            ax.set_xticklabels(['Left\n(100%)', '25%', '12.5%', '6.25%', '6.25%', '12.5%', '25%', 'Right\n(100%)'],
                               fontsize=8, rotation=45, ha='right')
        else:
            assert custom_inds is not None and custom_gradings is not None
            inds = custom_inds
            ax.set_xticks(custom_inds)
            # keep grading levels to at most one place after decimal
            custom_labels = [f'{g:.3g}' for g in custom_gradings]
            ax.set_xticklabels(custom_labels, fontsize=8, rotation=0)
    else:
        inds = [0, 12, 14, 15, 17, 18, 20, 32]
        assert psychometrics.shape[1] == len(inds)
        ax.set_xticks(inds)
        ax.set_xticklabels(['-1\nLeft', '', '', '', '', '', '', '+1\nRight'],
                           fontsize=8, rotation=0) #, ha='right')

    n = int(len(psychometrics) / plot_every)
    colors = [colormap(i) for i in np.linspace(0, 1, n)]

    for i in range(n):
        ax.plot(inds, psychometrics[plot_every * i], color=colors[i],
                lw=1.2, alpha=0.85, label=f'{label_word} #{plot_every * i}')
        
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, zorder=0)  # chance line

    ax.set_ylabel('P(Right Choice)', fontsize=23)
    ax.set_ylim(0, 1)
    ax.set_title(title if title else 'Psychometric Curve', fontsize=23)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    # ax.tick_params(axis='y', labelsize=8.5, length=3, width=0.8)
    # ax.tick_params(axis='x', length=3, width=0.8)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)


    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1'], fontsize=21)

    # increase all tick sizes
    ax.tick_params(axis='both', labelsize=21, length=4, width=1.0)

    if use_colormap:
        sm = plt.cm.ScalarMappable(cmap=colormap,
                                   norm=plt.Normalize(vmin=0., vmax=psychometrics.shape[0]))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.08)
        cbar.set_label(f'{label_word} #', fontsize=21)

        # colorbar ticks
        # upper = 5 * ((psychometrics.shape[0]) // 5)
        # cbar.set_ticks(np.arange(5, upper+1, 5))
        # # cbar.set_ticklabels(['5', '10', '15', '20'])
        cbar.set_ticks([0, psychometrics.shape[0]])
        cbar.set_ticklabels(['1', f'{psychometrics.shape[0]}'])
        cbar.ax.tick_params(labelsize=20)
    else:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, frameon=False)


    plt.tight_layout()
    
    if filepath is not None:
        plt.savefig(f'{filepath}/session_psychometrics.pdf', bbox_inches='tight')
    else:
        plt.show()


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


def plot_sweep_results(results_dict, threshold=0.1, xlim=5_000):
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

    fig, axs = plt.subplots(4, 1, figsize=(8, 20))
    plt.xlabel('Iterations', fontsize=18)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)

    axs[0].set_ylabel('Accuracy', fontsize=18)
    axs[1].set_ylabel('Slope Difference (R - L)', fontsize=18)
    axs[2].set_ylabel('Bias', fontsize=18)
    axs[3].set_ylabel('Stim DA Bias', fontsize=18)

    axs[0].set_xlim(0, 40)
    axs[1].set_xlim(0, xlim)
    axs[2].set_xlim(0, xlim)
    axs[3].set_xlim(0, xlim)

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
        axs[1].set_ylim(-0.8, 0.8)

        axs[2].plot(biases[i, :trials], color=color, alpha=0.5)
        axs[2].plot(trials-1, biases[i, trials-1], marker='o', color='cadetblue', alpha=0.5)
        axs[2].set_ylim(-0.5, 0.5)

        axs[3].plot(stim_DA_biases[i, :trials], color=color, alpha=0.5)
        axs[3].plot(trials-1, stim_DA_biases[i, trials-1], marker='o', color='cadetblue', alpha=0.5)
        axs[3].set_ylim(-0.5, 0.5)

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


def compute_two_layer_flattened_net_params(w):
    # w is T x 9 
    W1_diag = w[:, :3]          # shape (T, 3)
    W2 = w[:, 3:].reshape(-1, 2, 3)  # shape (T, 2, 3)
    
    # Multiply columns of W2 by diagonal entries of W1
    result = W2 * W1_diag[:, None, :]   # shape (T, 2, 3)
    
    # Flatten last two dims
    return result.reshape(-1, 6)        # shape (T, 6)


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

def visualize_weight_trajectories(depth, ws, filename=None):
    '''
    Arguments:
    depth  :  depth of network ('two-layer' or 'one-layer')
    ws     :  weight trajectories Pytree with keys 'W1_diag' and 'W2' and first dimension time T
    '''
    w_flat = flatten_params(ws)

    nrows = 3
    if depth == 'one-layer':
        nrows = 2
    
    fig, axs = plt.subplots(nrows, 3, figsize=(8, 2*nrows), sharex=True, sharey=True)

    j = 0
    for ax in axs.ravel():
        ax.plot(w_flat[:, j], lw=2.5, color='green', linestyle='dashed')
        ax.axhline(0, lw=2., color='gray', linestyle=':')
        j += 1

    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}')

    return


def compute_AIC(nll, num_params):
    return 2 * nll + 2 * num_params 



def stratified_quantile_split_indices(data, K, n_bins=10, random_state=None):
    """
    Stratified split of 1D data into train/test using quantile bins.
    Returns indices so you can apply them to multiple arrays.

    Args:
        data (list or np.array): Values between -1 and 1.
        K (int): Number of training samples.
        n_bins (int): Number of quantile bins to stratify over.
        random_state (int or None): Random seed.

    Returns:
        train_idx (np.array), test_idx (np.array)
    """
    rng = np.random.default_rng(random_state)
    data = np.array(data)
    N = len(data)

    # Compute quantile bin edges
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)

    # Assign each point to a bin
    bin_ids = np.digitize(data, bin_edges[1:-1])

    # Desired ratio
    train_ratio = 1. * K / N

    train_idx, test_idx = [], []

    # Stratified sampling within each bin
    for b in range(n_bins):
        bin_indices = np.where(bin_ids == b)[0]
        rng.shuffle(bin_indices)

        n_train_bin = int(round(train_ratio * len(bin_indices)))
        train_idx.extend(bin_indices[:n_train_bin])
        test_idx.extend(bin_indices[n_train_bin:])

    # Fix rounding mismatches
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    train_idx = np.array(train_idx)[:K]
    test_idx = np.array(test_idx)[:N - K]

    return train_idx, test_idx

