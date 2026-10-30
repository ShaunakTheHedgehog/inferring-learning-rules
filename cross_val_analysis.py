import os
import pickle as pkl
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from globals import MODELS, ALL_WITTEN_LEARNERS, EXPERT_THRESHOLD, NUM_FINAL_SESSIONS
from mouse_inference_data_analysis import check_expert_status, make_jitterplot

ALL_MODELS = ['RF_1', 'RF_b_1', '1L', '1L_sat', '2L', '2L_sat', '2NL']
NO_SAT_MODELS = ['RF_1', 'RF_b_1', '1L', '2L', '2NL']
NN_SAT_MODELS = ['1L', '1L_sat', '2L', '2L_sat', '2NL', '2NL_sat']

SMALL_SAT_SET = ['1L_sat', '2L', '2L_sat', '2NL']
MODEL_TO_SAT = {'1L': False, '1L_sat': True, '2L': False, '2L_sat': True, '2NL': False, '2NL_sat': True}
MODEL_TO_MODEL_TYPE = {'1L': '1L', '1L_sat': '1L', '2L': '2L', '2L_sat': '2L', '2NL': '2NL', '2NL_sat': '2NL'}


def load_cv_results(dataset_name, mouse_id, model, fold, results_type='test_nll', foldseed=0, data_root='.', sat_inputs=False):
    """Load the per-mouse behavioral dict (same file as mouse_inference_data_analysis.load_data)."""
    sat_tag = ""
    if sat_inputs:
        sat_tag = "_sat_inputs"
    path = os.path.join(data_root, f'{dataset_name}_cross_val_results{sat_tag}',
                       f'{model}', f'mouse_{mouse_id}', f'fold_{fold}_foldseed_{foldseed}.pkl')
    with open(path, 'rb') as f:
        results = pkl.load(f)

    if not results['converged']:
        print(f'Failure to converge! {mouse_id}, {model}, {fold}')

    return results[results_type]

def get_mouse_nll_comparison(nll_dict, mouse_id, models_used=ALL_MODELS, base_idx=-1, sign_flip=1):
    assert sign_flip in [1, -1]

    # get total number of trials for this mouse
    total_trials = sum(len(fold_nlls) for fold_nlls in nll_dict[mouse_id].values())

    model_nll_dict = nll_dict[mouse_id]
    all_fold_results = [] 
    stats = {}
    for i, model in enumerate(models_used):
        fold_results = np.asarray(model_nll_dict[model])
        all_fold_results.append(fold_results)

    all_fold_results = np.asarray(all_fold_results)
    baseline = all_fold_results[base_idx].reshape(1, -1) if base_idx is not None else 0
    if base_idx is None:
        assert sign_flip == 1, "If base_idx is None, sign_flip must be 1."
    
    diffs = sign_flip * (all_fold_results - baseline)

    for i in range(len(models_used)):
        model_diffs = diffs[i]
        stats[models_used[i]] = {'min': model_diffs.min().item(), 'mean': model_diffs.mean().item(), 
                                'max': model_diffs.max().item(), 'median': np.median(model_diffs).item(),
                                'std': np.std(model_diffs, ddof=1).item()}

    return stats, model_diffs, all_fold_results




if __name__ == "__main__":
    dataset_name = 'witten'
    K = 5

    path = os.path.join(f'{dataset_name}_data_GD_analysis_final',
                            f'{dataset_name}_mouse_data.pkl')
    with open(path, 'rb') as f:
        witten_data = pkl.load(f)

    mouse_ids = sorted(witten_data.keys())


    # nll_dict_wsat = {}
    # for mouse_id in mouse_ids:
    #     nll_dict_wsat[mouse_id] = {}
    #     for i, model in enumerate(NN_SAT_MODELS):
    #         fold_nlls = [load_cv_results(dataset_name, mouse_id, MODEL_TO_MODEL_TYPE[model], fold, sat_inputs=MODEL_TO_SAT[model]) for fold in range(K)]
    #         nll_dict_wsat[mouse_id][model] = fold_nlls

    # with open('mouse_test_nll_dict_wSAT.pkl', 'wb') as f:
    #     pkl.dump(nll_dict_wsat, f)

    # print(1./0)

    save_path = f'mouse_test_nll_dict_wSAT.pkl'
    with open(save_path, 'rb') as f:
        nll_dict = pkl.load(f)

    # print(nll_dict)

    mouse_ids = ALL_WITTEN_LEARNERS
    model_names = ['1L', '2L', '2NL']
    num_trials_by_mouse = []
    expert_idxs = []

    for i, mouse_id in enumerate(mouse_ids):
        num_trials_by_mouse.append(witten_data[mouse_id]['inputs'].shape[0])

        expert = check_expert_status(witten_data[mouse_id], threshold=EXPERT_THRESHOLD, num_final_sessions=NUM_FINAL_SESSIONS)
        if expert:
            expert_idxs.append(i)
    print(num_trials_by_mouse) 

    mean_nlls = np.zeros((len(mouse_ids), len(model_names)))
    for i, mouse_id in enumerate(mouse_ids):
        for j, model in enumerate(model_names):
            mean_nlls[i, j] = np.mean(nll_dict[mouse_id][model])

        print(f'{mouse_id}: {mean_nlls[i]}')

    make_jitterplot(mean_nlls, expert_idxs, save_path=f'witten_data_GD_analysis_final/aggregate_results/CV_nll_jitterplot_NEW.pdf', plot_type=None, ylabel=r'Mean $\Delta$NLL', plot_sem=True)

    print(np.argmin(mean_nlls, axis=1))

    expert_nlls = mean_nlls[expert_idxs]
    print(np.argmin(expert_nlls, axis=1))

    print(1./0)
    
    # print(nll_dict[35])
    # print(1./0)
    # save_path = 'mouse_test_nll_dict_full.pkl'
    # with open(save_path, 'rb') as f:
    #     nll_dict = pkl.load(f)

    # mouse_id = 35
    # means_RF_1 = 0
    # means_RF_b_1 = 0
    # means_1L = 0
    # means_2L = 0
    # means_2NL = 0
    # means_1L_sat = 0
    # means_2L_sat = 0
    
    # for mouse_id in mouse_ids:
    #     # stats, diffs, all_results = get_mouse_nll_comparison(nll_dict, mouse_id)

    #     og_stats, _, _ = get_mouse_nll_comparison(nll_dict, mouse_id, models_used=NN_SAT_MODELS, base_idx=None, sign_flip=1)

    #     is_expert = check_expert_status(witten_data[mouse_id])

    #     if is_expert or True:
    #         if og_stats['1L_sat']['mean'] > 0:
    #             means_1L_sat += 1
    #         if og_stats['2L_sat']['mean'] > 0:
    #             means_2L_sat += 1
    #         # if og_stats['1L']['mean'] > 0:
    #         #     means_1L += 1
    #         # if og_stats['2L']['mean'] > 0:
    #         #     means_2L += 1
    #         if og_stats['2NL']['mean'] > 0:
    #             means_2NL += 1
                
    #     # if not is_expert:
    #     #     continue 
    #     # else:
    #     #     num_experts += 1
    #     print(f'{mouse_id}, expert = {is_expert}:')
    #     # print(stats)
    #     print(og_stats)
    #     print('\n')
    # # print(means_1L_sat)
    # # print(means_2L_sat)
    # # print(means_1L)
    # # print(means_2L)
    # # print(means_2NL)


    # # mean_diffs = diffs.mean(-1)
    # # std_diffs = np.std(diffs, axis=-1, ddof=1)
    # # print(mean_diffs, std_diffs)


    
