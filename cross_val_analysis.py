import os
import pickle as pkl
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from globals import MODELS
from mouse_inference_data_analysis import check_expert_status

def load_cv_results(dataset_name, mouse_id, model, fold, results_type='test_nll', foldseed=0, data_root='.'):
    """Load the per-mouse behavioral dict (same file as mouse_inference_data_analysis.load_data)."""
    path = os.path.join(data_root, f'{dataset_name}_cross_val_results',
                       f'{model}', f'mouse_{mouse_id}', f'fold_{fold}_foldseed_{foldseed}.pkl')
    with open(path, 'rb') as f:
        results = pkl.load(f)

    return results[results_type]

def get_mouse_nll_comparison(nll_dict, mouse_id):
    model_nll_dict = nll_dict[mouse_id]
    means = np.zeros(len(MODELS))
    stds = np.zeros(len(MODELS))
    all_fold_results = [] 
    for i, model in enumerate(MODELS):
        fold_results = np.asarray(model_nll_dict[model])
        all_fold_results.append(fold_results)
        means[i] = fold_results.mean()
        stds[i] = np.std(fold_results, ddof=1)

    all_fold_results = np.asarray(all_fold_results)
    diffs = np.zeros((2, all_fold_results.shape[1]))
    diffs[0] = all_fold_results[0] - all_fold_results[2]
    diffs[1] = all_fold_results[1] - all_fold_results[2]

    return means, stds, all_fold_results, diffs




if __name__ == "__main__":
    dataset_name = 'witten'
    K = 5

    path = os.path.join(f'{dataset_name}_data_GD_analysis_final',
                            f'{dataset_name}_mouse_data.pkl')
    with open(path, 'rb') as f:
        witten_data = pkl.load(f)

    mouse_ids = sorted(witten_data.keys())

    save_path = f'{dataset_name}_cross_val_results/mouse_test_nll_dict.pkl'
    with open(save_path, 'rb') as f:
        nll_dict = pkl.load(f)

    mouse_id = 35
    nln_wins_1L = 0
    nln_wins_2L = 0
    num_experts = 0
    for mouse_id in mouse_ids:
        means, stds, all_results, diffs = get_mouse_nll_comparison(nll_dict, mouse_id)
        is_expert = check_expert_status(witten_data[mouse_id])
        if not is_expert:
            continue 
        else:
            num_experts += 1
        print(f'{mouse_id}, expert = {is_expert}: {diffs} \n')
        nln_wins_1L += int(np.all(diffs[0] > 0))
        nln_wins_2L += int(np.all(diffs[1] > 0))

    print(num_experts)
    print(nln_wins_1L)
    print(nln_wins_2L)

    # mean_diffs = diffs.mean(-1)
    # std_diffs = np.std(diffs, axis=-1, ddof=1)
    # print(mean_diffs, std_diffs)


    
