
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
from utils import *
from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from shallow_linear_net import shallow_net, shallow_net_params, fit_shallow_model
import metric_GD_network

def simple_sweep(model_type, num_models, num_sessions, num_samples_per_session, inputs, targets,
                 zero_contrast_mask, key, loss_type='tutor_executor', save_data=True, num_grading_levels=2,
                 zero_contrast_fraction=0.04, expert_threshold=0.7, accs_iter_threshold=34, nonneg_constraint=True): # 34 <--> 8500 iters
    '''
    Train multiple seeds of a shallow linear network on the same binary decision-making task,
    using one of two RL-based loss functions.

    Arguments:
    model_type  :   string indicating which model to train ('shallow' or 'deep')
    num_models  :   number of models to train
    num_sessions:   number of repeated full sessions to run
    num_samples_per_session :   number of training samples per session
    inputs      :   3-dimensional inputs (const, R, L), with shape (num_trials, 3)
    targets     :   1-dimensional outputs (0/R or 1/L), with shape (num_trials,)
    zero_contrast_mask : a binary mask with '1's for all non-zero contrast trials and '0's otherwise
    key         :   random number generator key
    loss_type   :   string indicating which loss function to use

    Returns:
    results_dict      :   dictionary of results for all models
    '''
    num_trials = num_sessions * num_samples_per_session
    assert loss_type in ['tutor_executor', 'single_loss_gd']
    assert model_type in ['shallow', 'deep']
    assert inputs.shape[0] == targets.shape[0]
    assert inputs.shape[0] == num_trials
    assert inputs.shape[1] == 3

    z = zero_contrast_fraction

    full_results_dict = {'losses' : [], 'rewards' : [], 'accs' : [], 'outcome_DA_signals' : [], 'stim_DA_signals' : [],
                         'biases' : [], 'stim_DA_biases' : [], 'psychometrics' : [], 'slope_diffs' : [],
                         'model_agnostic_biases' : [], 'partial_model_agnostic_biases' : [], 'model_based_biases' : [],
                         'partial_model_based_biases' : [], 'eff_num_trials' : []}

    for i in range(num_models):
        _, key, skey, sskey = jr.split(key, 4)

        params = None
        if model_type == 'shallow':
            params, (beta, lr) = shallow_net_params(skey)
        else:
            if loss_type == 'tutor_executor':
                params, (beta, lr) = random_init_params(skey, 3, 3, 2, default_tutor_executor=True)
            else:
                params, (beta, lr) = random_init_params(skey, 3, 3, 2, default_single_loss=True)

        print(f'\nTraining Model #{i+1}:')
        if model_type == 'shallow':
            params, losses, results_dict = fit_shallow_model(params, num_trials, inputs, targets, lr, beta, sskey, loss_type=loss_type,
                                                             end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels, nonneg_constraint=nonneg_constraint)
        else:
            params, losses, results_dict = fit_deep_net(params, num_trials, inputs, targets, lr, beta, sskey, loss_type=loss_type,
                                               end_condition=True, zero_contrast_mask=mask, num_sessions=num_sessions, num_grading_levels=num_grading_levels, nonneg_constraint=nonneg_constraint)


        losses = jnp.array(results_dict['losses'])
        rewards = jnp.array(results_dict['rewards'])
        rightward_statistics = results_dict['rightward_statistics']
        partial_rightward_statistics = results_dict['partial_rightward_statistics']

        accs = calculate_session_accs(rewards, num_samples_per_session, zero_contrast_mask)
        print(f'\nMean Accuracy by Trial: \n{accs}')
        if accs[:accs_iter_threshold].max() < expert_threshold:
            print('Failed to reach expert status!')
            continue

        eff_num_trials = results_dict['eff_num_trials']

        model_agnostic_bias = calculate_model_agnostic_bias(rightward_statistics, z)
        partial_agnostic_bias = calculate_model_agnostic_bias(partial_rightward_statistics, z)
        # print(f'\nModel-agnostic bias: {model_agnostic_bias}')
        print(f'\nModel-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

        model_based_bias = calculate_model_based_bias(results_dict['psychometrics'])
        partial_model_based_bias = calculate_model_based_bias(results_dict['psychometrics'][:eff_num_trials])
        # print(f'\nModel-based bias: {model_based_bias}')
        print(f'\nModel-based bias up to {eff_num_trials} iterations: {partial_model_based_bias}')

        full_results_dict['losses'].append(losses)
        full_results_dict['rewards'].append(rewards)
        full_results_dict['accs'].append(accs)
        full_results_dict['outcome_DA_signals'].append(results_dict['outcome_DA_signals'])
        full_results_dict['stim_DA_signals'].append(results_dict['stim_DA_signals'])
        full_results_dict['biases'].append(results_dict['biases'])
        full_results_dict['stim_DA_biases'].append(results_dict['stim_DA_biases'])
        full_results_dict['psychometrics'].append(results_dict['psychometrics'])
        full_results_dict['slope_diffs'].append(results_dict['slope_diffs'])
        full_results_dict['eff_num_trials'].append(eff_num_trials)

        full_results_dict['model_agnostic_biases'].append(model_agnostic_bias)
        full_results_dict['partial_model_agnostic_biases'].append(partial_agnostic_bias)
        full_results_dict['model_based_biases'].append(model_based_bias)
        full_results_dict['partial_model_based_biases'].append(partial_model_based_bias)

        print('\n\n')

    if save_data:
        filename = f'{model_type}_net_results/{model_type}net_{loss_type}_results_dict_nummodels{num_models}_numsessions{num_sessions}_numtrialspersession{num_samples_per_session}_numgradinglevels{num_grading_levels}_zerocontrastlevel{z}_nonnegconstraint{nonneg_constraint}.pkl'
        with open(filename, 'wb') as file:
            pkl.dump(full_results_dict, file)
        # np.save(f'NN_model_results/{model_type}net_{loss_type}_results_dict_nummodels{num_models}_numsessions{num_sessions}_numtrialspersession{num_samples_per_session}_lr{lr}_beta{beta}', full_results_dict)

    return full_results_dict


def metric_GD_model_sweep(model_type, num_models, num_sessions, num_samples_per_session, inputs, targets,
                          zero_contrast_mask, key, save_data=True, num_grading_levels=4,
                          zero_contrast_fraction=0.04, expert_threshold=0.7, accs_iter_threshold=34, 
                          weights_nonneg_constraint=False, training_nonneg_constraint=False, additional_info=''): # 34 <--> 8500 iters
    '''
    Train multiple seeds of a metric GD-based linear network on the same binary decision-making task,
    using the total squared error loss / RPE.

    Arguments:
    model_type  :   string indicating which model to train ('two-layer' or 'one-layer')
    num_models  :   number of models to train
    num_sessions:   number of repeated full sessions to run
    num_samples_per_session :   number of training samples per session
    inputs      :   3-dimensional inputs (const, R, L), with shape (num_trials, 3)
    targets     :   1-dimensional outputs (0/R or 1/L), with shape (num_trials,)
    zero_contrast_mask : a binary mask with '1's for all non-zero contrast trials and '0's otherwise
    key         :   random number generator key

    Returns:
    results_dict      :   dictionary of results for all models
    '''

    num_trials = num_sessions * num_samples_per_session
    assert model_type in ['two-layer', 'one-layer']
    assert inputs.shape[0] == targets.shape[0]
    assert inputs.shape[0] == num_trials
    assert inputs.shape[1] == 3

    z = zero_contrast_fraction

    full_results_dict = {'losses' : [], 'rewards' : [], 'accs' : [], 'outcome_DA_signals' : [], 'stim_DA_signals' : [],
                         'biases' : [], 'stim_DA_biases' : [], 'psychometrics' : [], 'slope_diffs' : [],
                         'model_agnostic_biases' : [], 'partial_model_agnostic_biases' : [], 'model_based_biases' : [],
                         'partial_model_based_biases' : [], 'eff_num_trials' : []}

    for i in range(num_models):
        _, key = jr.split(key, 2)
        skeys = jr.split(key, 3)

        params, beta, lrs = metric_GD_network.random_init_params(skeys[1], model_type=model_type, enforce_nonnegativity=weights_nonneg_constraint)

        print(f'\nTraining Model #{i+1}:')
        params, losses, results_dict = metric_GD_network.fit_metric_GD_net(model_type, params, num_trials=num_trials, num_sessions=num_sessions,
                                                     inputs=inputs, targets=targets, lrs=lrs, beta=beta,
                                                     zero_contrast_mask=zero_contrast_mask, key=skeys[2],
                                                     nonneg_constraint=training_nonneg_constraint, end_condition=True, num_grading_levels=num_grading_levels)


        losses = jnp.array(results_dict['losses'])
        rewards = jnp.array(results_dict['rewards'])
        rightward_statistics = results_dict['rightward_statistics']
        partial_rightward_statistics = results_dict['partial_rightward_statistics']

        accs = calculate_session_accs(rewards, num_samples_per_session, zero_contrast_mask)
        print(f'\nMean Accuracy by Trial: \n{accs}')
        if accs[:accs_iter_threshold].max() < expert_threshold:
            print('Failed to reach expert status!')
            continue

        eff_num_trials = results_dict['eff_num_trials']

        model_agnostic_bias = calculate_model_agnostic_bias(rightward_statistics, z)
        partial_agnostic_bias = calculate_model_agnostic_bias(partial_rightward_statistics, z)
        # print(f'\nModel-agnostic bias: {model_agnostic_bias}')
        print(f'\nModel-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')

        model_based_bias = calculate_model_based_bias(results_dict['psychometrics'])
        partial_model_based_bias = calculate_model_based_bias(results_dict['psychometrics'][:eff_num_trials])
        # print(f'\nModel-based bias: {model_based_bias}')
        print(f'\nModel-based bias up to {eff_num_trials} iterations: {partial_model_based_bias}')

        full_results_dict['losses'].append(losses)
        full_results_dict['rewards'].append(rewards)
        full_results_dict['accs'].append(accs)
        full_results_dict['outcome_DA_signals'].append(results_dict['outcome_DA_signals'])
        full_results_dict['stim_DA_signals'].append(results_dict['stim_DA_signals'])
        full_results_dict['biases'].append(results_dict['biases'])
        full_results_dict['stim_DA_biases'].append(results_dict['stim_DA_biases'])
        full_results_dict['psychometrics'].append(results_dict['psychometrics'])
        full_results_dict['slope_diffs'].append(results_dict['slope_diffs'])
        full_results_dict['eff_num_trials'].append(eff_num_trials)

        full_results_dict['model_agnostic_biases'].append(model_agnostic_bias)
        full_results_dict['partial_model_agnostic_biases'].append(partial_agnostic_bias)
        full_results_dict['model_based_biases'].append(model_based_bias)
        full_results_dict['partial_model_based_biases'].append(partial_model_based_bias)

        print('\n\n')

    if save_data:
        filename = f'metric_GD_data/{model_type}_net_results_dict_nummodels{num_models}_numsessions{num_sessions}_numtrialspersession{num_samples_per_session}_numgradinglevels{num_grading_levels}_zerocontrastlevel{z}_nonnegconstraint{nonneg_constraint}_{additional_info}.pkl'
        with open(filename, 'wb') as file:
            pkl.dump(full_results_dict, file)

    return full_results_dict




if __name__=="__main__":
    key = jr.PRNGKey(102)    # used 31 before
    _, key = jr.split(key, 2)

    skeys = jr.split(key, 5)

    num_models = 40

    num_sessions = 40

    expert_threshold = 0.7
    model_type = 'two-layer'
    # loss_type = 'tutor_executor'

    num_grading_levels = 4
    num_samples_per_grading = 30
    num_zero_contrast = 10
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session
    nonneg_constraint = False

    inputs, targets, mask = generate_randomized_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, skeys[1])

    results_dict = metric_GD_model_sweep(model_type, num_models, num_sessions, num_samples_per_session,
                                         inputs, targets, mask, skeys[2], save_data=True, num_grading_levels=num_grading_levels,
                                         zero_contrast_fraction=z, weights_nonneg_constraint=True, training_nonneg_constraint=False, additional_info='nonneg_init_weights')

    # with open('metric_GD_data/two-layer_net_results_dict_nummodels40_numsessions40_numtrialspersession250_numgradinglevels4_zerocontrastlevel0.04_nonnegconstraintFalse_run2.pkl', 'rb') as file:
    #     results_dict = pkl.load(file)
    
    eff_num_trials = np.asarray(results_dict['eff_num_trials'])
    print(f'\nEffective number of trials: {eff_num_trials}')
    num_models = len(eff_num_trials)

    slope_diffs = results_dict['slope_diffs']
    early_slope_diffs = [slope_diffs[i][1_000] for i in range(num_models)]
    late_slope_diffs = [slope_diffs[i][eff_num_trials[i]-1] for i in range(num_models)]
    print(f'Early slope diffs: {early_slope_diffs}')
    print(f'Late slope diffs: {late_slope_diffs}')

    biases = np.asarray(results_dict['biases'])
    print(biases.shape)
    early_biases = [biases[i, 1_000] for i in range(num_models)]
    late_biases = [biases[i, eff_num_trials[i]-1] for i in range(num_models)]


    stim_DA_biases = np.asarray(results_dict['stim_DA_biases'])
    print(stim_DA_biases.shape)
    early_stim_DA_biases = [stim_DA_biases[i, 1_000] for i in range(num_models)]
    late_stim_DA_biases = [stim_DA_biases[i, eff_num_trials[i]-1] for i in range(num_models)]


    pma_biases = np.asarray(results_dict['partial_model_agnostic_biases'])
    print(pma_biases.shape)

    pmb_biases = np.asarray(results_dict['partial_model_based_biases'])
    print(pmb_biases.shape)

    # show_histogram(early_slope_diffs, bins=20, title='R-L Slope (Early -- Iter 1,000)')
    # show_histogram(late_slope_diffs, bins=20, title='R-L Slope (Late)')

    # show_histogram(early_stim_DA_biases, bins=20, title='Stim DA Signal (Early -- Iter 1,000)')
    # show_histogram(late_stim_DA_biases, bins=20, title='Stim DA Signal (Late)')

    # show_histogram(early_biases, bins=20, title='Bias (Early -- Iter 1,000)')
    # show_histogram(late_biases, bins=20, title='Bias (Late)')
    # # show_histogram(biases[:, -1], bins=35, title='Bias (Iter 10,000)')

    # show_histogram(pma_biases, bins=20, title='Model-agnostic Bias (Partial)')

    # show_histogram(pmb_biases, bins=20, title='Model-based Bias (Partial)')

    plot_sweep_results(results_dict, threshold=0.15)
