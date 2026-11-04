import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pickle as pkl

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import calculate_simple_mouse_ma_bias, get_zero_contrast_mask
from utils import calculate_batched_quantity_per_session, calculate_running_avg_quantity
from globals import NUM_FINAL_SESSIONS, DATASET_TO_CHOICE_KEYWORD, ALL_WITTEN_LEARNERS

# Shared figure styling so both plots read the same
LABEL_FS = 22    # axis labels
TICK_FS = 18     # tick labels
LEGEND_FS = 18   # legend entries
TRIAL_XTICKS = [4000, 8000, 12000]


def style_axes(ax, ylabel, xlabel='Trial', xticks=TRIAL_XTICKS):
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x:,}' for x in xticks])
    ax.tick_params(axis='both', labelsize=TICK_FS, length=6, width=1.2)
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(1.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#dddddd')
    ax.set_axisbelow(True)


def load_data(dataset_name, keys=None):
    assert dataset_name in ['witten', 'saxe'], f'Invalid dataset name: {dataset_name}'
    with open(f'{dataset_name}_data_GD_analysis_CLEANED/{dataset_name}_mouse_data_CLEANED.pkl', 'rb') as f:
        data = pkl.load(f)

    print(f"Loaded data for {len(data)} mice from dataset '{dataset_name}'")
    
    if keys is not None:
        data = {k: v for k, v in data.items() if k in keys}

    return data

def check_expert_status(mouse_dict, threshold=0.7, num_final_sessions=NUM_FINAL_SESSIONS):
    mask = get_zero_contrast_mask(mouse_dict['inputs'])
    rewards = mouse_dict['rewards']
        
    trials_per_session = mouse_dict['trials_per_session']
    session_accs = calculate_batched_quantity_per_session(rewards, trials_per_session, mask=mask)
    
    final_accs = session_accs[-num_final_sessions:]
    avg_final_acc = final_accs.mean()
    return avg_final_acc >= threshold


def compute_max_trials(dataset_name):
    data = load_data(dataset_name)

    max_trials = 0

    for mouse_id in data.keys():
        mouse_data = data[mouse_id]
        ntrials = len(mouse_data['inputs'])
        if ntrials > max_trials:
            max_trials = ntrials
    
    return max_trials


def plot_all_accs(dataset_name, window=200, stride=50, save_path=None, dpi=300, upper_xlim=12_000, keys=None):
    assert dataset_name in ['witten', 'saxe'], f'Invalid dataset name: {dataset_name}'

    data = load_data(dataset_name, keys=keys)
    max_trials = compute_max_trials(dataset_name)

    fig, ax = plt.subplots(figsize=(7, 4))
    # ax.set_title(f'Running Accuracy Across Trials', fontsize=20)

    color = "#D2B8B8FF"
    avg_color = "#0A0A0A"
    sem_color = "#B0B2D8"

    trials = None
    traces = []

    for mouse_id in data.keys():
        mouse_data = data[mouse_id]

        mask = get_zero_contrast_mask(mouse_data['inputs'])
        rewards = mouse_data['rewards']
        outputs = mouse_data[DATASET_TO_CHOICE_KEYWORD[dataset_name]]

        running_avg_quantity = calculate_running_avg_quantity(rewards, window=window, mask=mask)
        padded_running_avg_quantity = np.full(max_trials, np.nan)
        padded_running_avg_quantity[:len(running_avg_quantity)] = running_avg_quantity
        traces.append(padded_running_avg_quantity)

        expert = check_expert_status(mouse_data)
        trials = np.arange(1, len(rewards) + 1)

        ax.plot(trials[::stride], running_avg_quantity[::stride],
                color=color,
                linewidth=0.8)
    
    traces = jnp.array(traces) 
    mean_trace = jnp.nanmean(traces, axis=0)

    n_valid = jnp.sum(~jnp.isnan(traces), axis=0)
    sem_trace = jnp.nanstd(traces, axis=0) / jnp.sqrt(n_valid)

    all_trials = jnp.arange(1, len(mean_trace)+1)

    ax.plot(all_trials[::stride], mean_trace[::stride],
                color=avg_color, 
                linewidth=2.5)
    
    ax.fill_between(all_trials[::stride], mean_trace[::stride] - sem_trace[::stride], mean_trace[::stride] + sem_trace[::stride],
                    color=sem_color, alpha=0.7, linewidth=0.)
    
    ax.set_xlim(right=upper_xlim)

    # Chance baseline
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.0, label='Chance (0.5)')

    style_axes(ax, 'Accuracy')
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(top=1.1)  # headroom so the top-left legend clears the traces

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=color, label='Each Mouse'),
        Patch(color=avg_color, label='Average')
        # plt.Line2D([0], [0], color='gray', linestyle='--', label='Chance (0.5)'),
    ], fontsize=LEGEND_FS, frameon=False,
       loc='upper left', borderaxespad=0.1, handlelength=1.5, labelspacing=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()



def plot_all_biases(dataset_name, window=200, stride=50, save_path=None, dpi=300, upper_xlim=12_000, bias_thresholds=[-0.2, 0.2], keys=None):
    assert dataset_name in ['witten', 'saxe'], f'Invalid dataset name: {dataset_name}'

    data = load_data(dataset_name, keys=keys)
    print(f"Loaded data for {len(data)} mice from dataset '{dataset_name}'")
    max_trials = compute_max_trials(dataset_name)

    fig, ax = plt.subplots(figsize=(7, 4))
    # ax.set_title(f'Running Bias Across Trials', fontsize=20)
    # ax.set_title(f'Running Bias Across Trials', fontsize=20, pad=17)  # default is ~6


    right_biased_color = "#1AB32CFF"
    balanced_color = "#E5801B"
    left_biased_color = "#A50BEC"

    left_threshold = bias_thresholds[0]
    right_threshold = bias_thresholds[1]

    bias_traces = {'right-biased': [], 'balanced': [], 'left-biased': []}
    bias_to_color = {'right-biased': right_biased_color, 'balanced': balanced_color, 'left-biased': left_biased_color}

    ma_biases = []

    for mouse_id in data.keys():
        mouse_data = data[mouse_id]

        mask = get_zero_contrast_mask(mouse_data['inputs'])
        outputs = mouse_data[DATASET_TO_CHOICE_KEYWORD[dataset_name]]

        ma_bias = calculate_simple_mouse_ma_bias(outputs)
        ma_biases.append(ma_bias)

        rights = (1 - 2*outputs)

        running_avg_quantity = calculate_running_avg_quantity(rights, window=window, mask=mask)
        padded_running_avg_quantity = np.full(max_trials, np.nan)
        padded_running_avg_quantity[:len(running_avg_quantity)] = running_avg_quantity

        indiv_color = None
        if ma_bias >= right_threshold:
            bias_traces['right-biased'].append(padded_running_avg_quantity)
            indiv_color = right_biased_color
        elif ma_bias <= left_threshold:
            bias_traces['left-biased'].append(padded_running_avg_quantity)
            indiv_color = left_biased_color
        else:
            bias_traces['balanced'].append(padded_running_avg_quantity)
            indiv_color = balanced_color

        trials = np.arange(1, len(outputs) + 1)

        ax.plot(trials[::stride], running_avg_quantity[::stride],
                color=indiv_color, alpha=0.25,
                linewidth=0.8)
    
    for bias_type in bias_traces.keys():
        color = bias_to_color[bias_type]
        traces = jnp.array(bias_traces[bias_type])
        print(len(traces))
        mean_trace = jnp.nanmean(traces, axis=0)

        n_valid = jnp.sum(~jnp.isnan(traces), axis=0)
        sem_trace = jnp.nanstd(traces, axis=0) / jnp.sqrt(n_valid)

        all_trials = jnp.arange(1, len(mean_trace)+1)

        ax.plot(all_trials[::stride], mean_trace[::stride],
                    color=color, 
                    linewidth=2.5)
        
        ax.fill_between(all_trials[::stride], mean_trace[::stride] - sem_trace[::stride], mean_trace[::stride] + sem_trace[::stride],
                        color=color, alpha=0.25, linewidth=0.)
        
        ax.set_xlim(right=upper_xlim)

    style_axes(ax, 'Bias')
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    # Legend
    # ax.legend(handles=[
    #     Patch(color=right_biased_color, label='Right-biased'),
    #     Patch(color=balanced_color, label='Balanced'),
    #     Patch(color=left_biased_color, label='Left-biased')
    #     # plt.Line2D([0], [0], color='gray', linestyle='--', label='Chance (0.5)'),
    # ], fontsize=12.5, frameon=False)
    legend_elements = [
        Line2D([0], [0], color=right_biased_color, label='Right-biased', lw=2.5),
        Line2D([0], [0], color=balanced_color, label='Balanced', lw=2.5),
        Line2D([0], [0], color=left_biased_color, label='Left-biased', lw=2.5)
        # plt.Line2D([0], [0], color='gray', linestyle='--', label='Chance (0.5)'),
    ]

    # Anchor the legend's bottom edge just above the axes so it never overlaps the data;
    # tight_layout / bbox_inches="tight" account for it because it belongs to the axes.
    ax.legend(
        handles=legend_elements,
        fontsize=LEGEND_FS, frameon=False,
        loc='lower center', bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_elements),
        handlelength=1.5, columnspacing=1.2, borderaxespad=0.,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


# biases = plot_all_biases('witten')
# plt.hist(biases, bins=100)
# plt.show()

dataset = 'witten' 
keys = ALL_WITTEN_LEARNERS 
save_path = f'raw_mouse_behavior/{dataset}_mouse_biases_CLEANED.pdf'
plot_all_biases(dataset, window=400, stride=200, save_path=save_path, bias_thresholds=[-0.15, 0.15], keys=keys)

save_path = f'raw_mouse_behavior/{dataset}_mouse_accs_CLEANED.pdf'
plot_all_accs(dataset, window=400, stride=200, save_path=save_path, dpi=300, upper_xlim=12_000, keys=keys)

