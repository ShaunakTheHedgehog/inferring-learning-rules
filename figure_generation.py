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

from mouse_inference_data_analysis import *
from utils import convert_inds_to_grading_levels, generate_per_session_psychometrics, vectorized_generate_per_session_psychometrics, extract_specific_grading_levels


def generate_single_mouse_model_psychometrics(dataset_name, top_k, specific_grading_levels,
                                              infer_inds=False):
    use_witten_scheme = (dataset_name == 'witten')
    if dataset_name == 'saxe':
        assert infer_inds, "Must infer inds since grading levels vary from mouse to mouse in Liebana et al. dataset."

    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=True)
    mouse_ids = list(all_mice_behav_dict.keys())
    for mouse_id in mouse_ids:
        mouse_dict = all_mice_behav_dict[mouse_id]
        inputs = mouse_dict['inputs']
        outputs = mouse_dict[DATASET_TO_CHOICE_KEYWORD[dataset_name]].astype(jnp.int32)
        ma_bias = calculate_simple_mouse_ma_bias(outputs)

        print(f'Mouse {mouse_id}')
        expert = check_expert_status(mouse_dict)
        print(f'Expert: {expert}')
        print(f'Bias: {ma_bias}')

        base_filepath = f'{dataset_name}_data_GD_analysis_final/single_trial_results'

        trials_per_session = mouse_dict['trials_per_session']
        print(f'Num Sessions: {len(trials_per_session)}\n')

        true_psychometrics, inds, _, _ = vectorized_generate_per_session_psychometrics(inputs, outputs, trials_per_session, 
                                                        num_grading_levels=NUM_GRADING_LEVELS, specific_grading_levels=specific_grading_levels)
        custom_inds, custom_gradings = None, None
        if infer_inds:
            custom_inds = inds.astype(int)
            custom_gradings = convert_inds_to_grading_levels(custom_inds)
            print(f'Custom Indices: {custom_inds}')
            print(f'Inferred Grading Levels: {custom_gradings}')
        
        true_filepath = f'{base_filepath}/true_mouse/mouse_{mouse_id}'
        plot_psychometrics(true_psychometrics, plot_every=1, use_colormap=True, filepath=None, 
                       title='True Mouse', label_word='Session', use_witten_scheme=use_witten_scheme, use_ibl_ticks=False,
                       custom_inds=custom_inds, custom_gradings=custom_gradings)
        


        for i, key in enumerate(MODEL_CLASS_INFO):
            model_name = MODEL_PLOT_NAMES[i]
            inference_model, depth, nln_name, num_params, color = MODEL_CLASS_INFO[key]

            model_outputs = np.load(f'{dataset_name}_data_GD_analysis_final/single_trial_results/{inference_model}_{nln_name}_{depth}_results/mouse_{mouse_id}/all_model_outputs.npy')
            best_model_outputs, k_best_model_outputs, _ = find_best_matching_model_outputs(outputs, model_outputs, top_k=top_k)

            avg_model_psychometrics = np.zeros_like(true_psychometrics)
            for i in range(top_k):
                model_psychometrics, _, _, _ = vectorized_generate_per_session_psychometrics(inputs, k_best_model_outputs[i], trials_per_session, 
                                                        num_grading_levels=NUM_GRADING_LEVELS, specific_grading_levels=specific_grading_levels)
                
                avg_model_psychometrics += (model_psychometrics / top_k)
                            
            model_filepath = f'{base_filepath}/{inference_model}_{nln_name}_{depth}_results/mouse_{mouse_id}'
            plot_psychometrics(avg_model_psychometrics, plot_every=1, use_colormap=True, filepath=None, 
                       title=model_name, label_word='Session', use_witten_scheme=use_witten_scheme, use_ibl_ticks=False,
                       custom_inds=custom_inds, custom_gradings=custom_gradings)
        


def visualize_model_evolution(dataset_name, mouse_ids, grading=0.125, custom_slope_ticks=[-2., 0., 2., 4.], slope_ylim=(-2.2, 4.2),
                              plot_slopes=True):
    base_filepath = f'{dataset_name}_data_GD_analysis_final/single_trial_results'
    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=True)

    for mouse_id in mouse_ids:
        mouse_dict = all_mice_behav_dict[mouse_id]
        print(mouse_dict.keys())
        inputs = mouse_dict['inputs']
        outputs = mouse_dict[DATASET_TO_CHOICE_KEYWORD[dataset_name]].astype(jnp.int32)
        
        trials_per_session = mouse_dict['trials_per_session']

        # all_vars = []
        all_bias_r2s = []
        all_small_slope_r2s = []
        for i, key in enumerate(MODEL_CLASS_INFO):
            model_name = MODEL_PLOT_NAMES[i]
            inference_model, depth, nln_name, num_params, color = MODEL_CLASS_INFO[key]

            all_model_outputs = np.load(f'{base_filepath}/{inference_model}_{nln_name}_{depth}_results/mouse_{mouse_id}/all_model_outputs.npy')

            true_biases, model_biases, vars = get_model_bias_evolution_from_outputs(outputs, all_model_outputs, trials_per_session, calculate_within_session=True)
            plot_model_trace_evolution_by_session(true_biases, model_biases, 'Bias', 'deterministic', color, title=model_name, filepath=f'{base_filepath}/{inference_model}_{nln_name}_{depth}_results/mouse_{mouse_id}')
            bias_r2s = compute_r2_distribution(model_biases, true_biases)
            all_bias_r2s.append(bias_r2s)

            # all_vars.append(vars)

            if plot_slopes:
                model_slopes_fixedgrad = calculate_model_slope_fixed_grading(grading, inputs, all_model_outputs, trials_per_session, num_grading_levels=NUM_GRADING_LEVELS)
                true_slopes_fixedgrad = calculate_slope_fixed_grading(grading, inputs, outputs, trials_per_session, num_grading_levels=NUM_GRADING_LEVELS)
                plot_model_trace_evolution_by_session(true_slopes_fixedgrad, model_slopes_fixedgrad, 'Slope Difference', 'deterministic', color, title=model_name, filepath=f'{base_filepath}/{inference_model}_{nln_name}_{depth}_results/mouse_{mouse_id}', 
                                                    fixed_grading=grading, custom_ticks=custom_slope_ticks, ymin=slope_ylim[0], ymax=slope_ylim[1])
                small_slope_r2s = compute_r2_distribution(model_slopes_fixedgrad, true_slopes_fixedgrad)
                all_small_slope_r2s.append(small_slope_r2s)


        visualize_r2_distributions(all_bias_r2s, MODEL_COLORS, filepath=f'{base_filepath}/r2_distributions/mouse_{mouse_id}_bias_traj_r2_distributions.pdf')
        
        if plot_slopes:
            visualize_r2_distributions(all_small_slope_r2s, MODEL_COLORS, filepath=f'{base_filepath}/r2_distributions/mouse_{mouse_id}_grading={grading}_contrast_slopediff_r2_distributions.pdf')
   


def visualize_reconstructed_accs(dataset_name, top_k, plot=True, save=True):
    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=True)
    mouse_ids = list(all_mice_behav_dict.keys())

    all_r2s = []
    expert_idxs = []
    for i, mouse_id in enumerate(mouse_ids):

        expert = check_expert_status(all_mice_behav_dict[mouse_id], threshold=EXPERT_THRESHOLD, num_final_sessions=NUM_FINAL_SESSIONS)
        if expert:
            expert_idxs.append(i)

        num_trials = len(all_mice_behav_dict[mouse_id]['inputs'])

        plot_filepath = f'{dataset_name}_data_GD_analysis_final/single_trial_results/running_accs/top{top_k}'
        running_accs_info, r2s, _, _ = get_reconstructed_running_accs(dataset_name, mouse_id,
                                        top_k, window=200, stride=50, sampling_type='deterministic',
                                        plot=plot, legend=False, save=save, plot_filepath=plot_filepath,
                                        custom_xticks=None, error_bars='std')
    
        all_r2s.append(r2s)
        
    all_r2s = np.asarray(all_r2s)    

    print(f'Num experts: {len(expert_idxs)} out of {len(mouse_ids)} total mice.')
    make_2D_scatterplot(all_r2s, 'two-layer nonlinear', 'one-layer linear', expert_idxs, save_path=f'{dataset_name}_data_GD_analysis_final/aggregate_results/reconstructed_accs/top{top_k}_R2_vs_onelayer.pdf', measure=r'$R^2$')
    make_2D_scatterplot(all_r2s, 'two-layer nonlinear', 'two-layer linear', expert_idxs, save_path=f'{dataset_name}_data_GD_analysis_final/aggregate_results/reconstructed_accs/top{top_k}_R2_vs_twolayer.pdf', measure=r'$R^2$')


def visualize_trialwise_accs(dataset_name, top_k, stage_length=2_000,window=100, stride=25, ymin=0.4):
    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=True)
    mouse_ids = list(all_mice_behav_dict.keys())

    rightbiased_experts = []
    leftbiased_experts = []
    balanced_experts = []

    biases = []
    x0s = []

    for i, mouse_id in enumerate(mouse_ids):

        expert = check_expert_status(all_mice_behav_dict[mouse_id], threshold=EXPERT_THRESHOLD, num_final_sessions=NUM_FINAL_SESSIONS)
        inputs = all_mice_behav_dict[mouse_id]['inputs']
        rewards = all_mice_behav_dict[mouse_id]['rewards']
        outputs = all_mice_behav_dict[mouse_id]['outputs'].astype(jnp.int32)
        ma_bias = calculate_simple_mouse_ma_bias(outputs) 
        biases.append(ma_bias)

        print(f'Mouse {mouse_id}, Bias: {ma_bias}, Expert: {expert}')
        
        if expert:
            if ma_bias > 0.2:
                rightbiased_experts.append(mouse_id)
            elif ma_bias < -0.2:
                leftbiased_experts.append(mouse_id)
            else:
                balanced_experts.append(mouse_id)

            trials_per_session = all_mice_behav_dict[mouse_id]['trials_per_session']
        
            get_stage_accs(dataset_name, mouse_id, 'early', stage_length=stage_length, top_k=top_k, window=window, stride=stride, 
                    sampling_type='deterministic', plot=True, save=True, save_label=f'mouse_{mouse_id}', trials_per_session=trials_per_session, ymin=ymin)

            get_stage_accs(dataset_name, mouse_id, 'middle', stage_length=stage_length, top_k=top_k, window=window, stride=stride, 
                        sampling_type='deterministic', plot=True, save=True, save_label=f'mouse_{mouse_id}', trials_per_session=trials_per_session, ymin=ymin)
            
            get_stage_accs(dataset_name, mouse_id, 'late', stage_length=stage_length, top_k=top_k, window=window, stride=stride, 
                        sampling_type='deterministic', plot=True, save=True, save_label=f'mouse_{mouse_id}', trials_per_session=trials_per_session, ymin=ymin)
            

            for stage in ['early', 'middle', 'late']:
                pearsons = get_pearson_r2_for_accs(dataset_name, mouse_id, stage, stage_length=stage_length, top_k=top_k, window=window, 
                                sampling_type='deterministic')
                # print(pearsons)
                filepath = f'{dataset_name}_data_GD_analysis_final/single_trial_results/running_accs/single_trial/mouse_{mouse_id}_pearson_corrs_{stage}_stage_length={stage_length}_top{top_k}_window={window}_stride={stride}.pdf'
                visualize_r2_distributions(pearsons, MODEL_COLORS, filepath=filepath, xlabel=f'Pearson Correlation')
                
        

    print(f'Num Right Biased: {len(rightbiased_experts)}')
    print(f'Num Left Biased: {len(leftbiased_experts)}')
    print(f'Num Balanced: {len(balanced_experts)}')


def generate_aggregate_model_statistics(dataset_name, remove_bad_idxs=True):
    generate_aggregate_model_metrics(dataset_name, remove_bad_idxs=remove_bad_idxs)

    aic_data, nll_data, expert_idxs = generate_model_nll_comparison(dataset_name, get_aic_per_trial=True, remove_bad_idxs=True)
    print(f'Number of expert mice: {len(expert_idxs)}')

    save_path = f'{dataset_name}_data_GD_analysis_final/aggregate_results/rel_aic_jitterplot.pdf'
    make_jitterplot(aic_data, expert_idxs, save_path=save_path, show_legend=False, plot_sem=True, ticksize=26, ylabel='',
                    custom_yticks=[0., 0.1, 0.2])



if __name__ == "__main__":

    dataset_name = 'witten'
    max_contrast = 1.
    all_mice_behav_dict = load_data(dataset_name, remove_bad_idxs=True)
    mouse_ids = list(all_mice_behav_dict.keys())
    top_k = 100

    # generate_single_mouse_model_psychometrics(dataset_name, top_k=100, specific_grading_levels=None,
    #                                             infer_inds=True)   
    # visualize_single_mouse_model_weights_and_aics(dataset_name, remove_bad_idxs=True)
    # visualize_model_evolution(dataset_name, mouse_ids=mouse_ids, grading=None, custom_slope_ticks=[-2., 0., 2., 4.], slope_ylim=(-2.2, 4.2), plot_slopes=False)
   
    # visualize_model_evolution('witten', mouse_ids=[36], grading=0.125, custom_slope_ticks=[-1., 0., 1., 2.], slope_ylim=(-1.5, 2.5))

    # visualize_reconstructed_accs('saxe', top_k=100, plot=True, save=True)
    # visualize_trialwise_accs('witten', top_k=100, ymin=0.4)

    # generate_aggregate_model_metrics('witten', plot_pc_idx1=2, plot_pc_idx2=3, remove_bad_idxs=True)
    # visualize_reconstructed_accs('witten', top_k=100, plot=False, save=False)


    # for mouse_id in mouse_ids:
    #     expert = check_expert_status(all_mice_behav_dict[mouse_id], threshold=EXPERT_THRESHOLD, num_final_sessions=NUM_FINAL_SESSIONS)
    #     print(f'Mouse {mouse_id}, Expert: {expert}')

    generate_aggregate_model_statistics('saxe', remove_bad_idxs=True)
