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
from utils import generate_multisession_graded_data, calculate_model_agnostic_bias, calculate_session_accs, flatten_params, turn_pytree_to_arr, realign_weight_trajectories, combine_batch_weights
from simple_rates_and_weights_inference import *


def single_trial_simulated_aic_test(true_learning_rule, true_model_type, key, session_info=(40, 4, 30, 10), 
                                    num_opt_seeds=15, fixed_alpha=None, fixed_beta=None, fixed_w0=None, add_two_layer_reg=True, one_layer_scale=0.1):
        
    assert true_learning_rule in ['gd', 'te']
    assert true_model_type in ['two-layer', 'one-layer']
    true_num_params = 11
    if true_model_type == 'one-layer':
        true_num_params = 8
    _, key, skey = jr.split(key, 3)

    skeys = jr.split(skey, 6)

    print(f'TRUE MODEL: a {true_model_type} network trained with the {true_learning_rule} learning rule.')

    # 4 columns, in order, are 2-layer GD, 2-layer TE, 1-layer GD, 1-layer TE
    model_types = ['two-layer', 'two-layer', 'one-layer', 'one-layer']
    test_learning_rules = ['gd', 'te', 'gd', 'te']

    nlls = np.zeros(5)

    # number of params in each model type
    ps = np.array([true_num_params, 11, 11, 8, 8])

    inferred_hyperparams_collection = np.zeros((4, 11))

    alpha_star, beta_star = None, None
    if fixed_alpha is not None:
        assert fixed_beta is not None 
        alpha_star = fixed_alpha
        beta_star = fixed_beta 
    else:
        alpha_star = jr.uniform(skeys[1], minval=0.002, maxval=0.02)
        beta_star = jr.uniform(skeys[2], minval=5.0, maxval=15.0)

    data, _ = generate_train_test_split(alpha_star, beta_star, true_learning_rule, true_model_type, session_info, skeys[3], 
                              num_train_sets=1, num_test_sets=0, vary_inputs=False, true_w0=fixed_w0)#, one_layer_scale=one_layer_scale)
    inputs, outputs, rewards, train_w0 = data[0]


    w0_arr = turn_pytree_to_arr(train_w0)
    alpha_beta_arr = np.log(np.array([alpha_star, beta_star]))
    true_hyperparams_arr = np.concatenate((alpha_beta_arr, w0_arr))

    true_nll = compute_nll(true_hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=true_learning_rule, 
                           num_alpha_blocks=1, model_type=true_model_type)

    # true_nll = compute_batched_nll(true_hyperparams_arr, inputs, outputs, rewards, hyperparams_is_list=True, learning_rule=true_learning_rule, model_type=true_model_type)
    nlls[0] = true_nll
    print(f'Training Data (ground truth) NLL with {true_learning_rule} as the true learning rule: {true_nll}')


    curr_key = skeys[5]
    for j in range(len(model_types)):
        model_type = model_types[j]
        test_learning_rule = test_learning_rules[j]
        _, curr_key = jr.split(curr_key, 2)

        two_layer_lambda = 0.
        if (model_type == 'two-layer') and (true_model_type == 'one-layer') and add_two_layer_reg:
            two_layer_lambda = 1./(2 * one_layer_scale)     # regularization for w1 and W2 from a Bayesian perspective
            print(f'Adding L2 regularization with ridge = {two_layer_lambda}')

        print(f'\nPerforming inference in a {model_type} network with the {test_learning_rule} learning rule...')
        num_alpha_blocks = 1
        _, min_nll, hyperparams_arr = run_randomized_optimization_procedure(inputs, outputs, rewards, curr_key, learning_rule=test_learning_rule, model_type=model_type,
                                                                            num_seeds=num_opt_seeds, max_iters=500, 
                                                                            tol=1e-4, method='optax',
                                                                            alpha_init_scale=0.1, beta_init_scale=1., w0_init_scale=0.1, 
                                                                            num_alpha_blocks=num_alpha_blocks, l2_lambda=two_layer_lambda, batched=False)
        nlls[j + 1] = min_nll
        inferred_hyperparams_collection[j, :len(hyperparams_arr)] = hyperparams_arr
        print(f'Inferred (training) NLL with {test_learning_rule} as the true learning rule: {min_nll}')

    AICs = 2 * nlls + 2 * ps

    return nlls, AICs, true_hyperparams_arr, inferred_hyperparams_collection



def multi_trial_simulated_aic_test(true_learning_rule, true_model_type, num_trials, key, session_info=(40, 4, 30, 10), 
                                   num_opt_seeds=15, fixed_alpha=None, fixed_beta=None, fixed_w0=None, save_data=True, 
                                   filename='single_trial_tests', add_two_layer_reg=True, one_layer_scale=0.1):
    assert true_learning_rule in ['gd', 'te']
    assert true_model_type in ['two-layer', 'one-layer']
    true_num_params = 11
    if true_model_type == 'one-layer':
        true_num_params = 8

    all_nlls = np.zeros((num_trials, 5))
    all_AICs = np.zeros((num_trials, 5))
    true_hyperparams_set = np.zeros((num_trials, true_num_params))
    inferred_hyperparams_set = np.zeros((num_trials, 4, 11))

    for i in range(num_trials):
        _, key = jr.split(key, 2)
        nlls, AICs, true_hyperparams_arr, inferred_hyperparams_collection = single_trial_simulated_aic_test(true_learning_rule, true_model_type, key, session_info=session_info,
                                                                                                            num_opt_seeds=num_opt_seeds, fixed_alpha=fixed_alpha, 
                                                                                                            fixed_beta=fixed_beta, fixed_w0=fixed_w0, add_two_layer_reg=add_two_layer_reg, one_layer_scale=one_layer_scale)
        
        all_nlls[i] = nlls 
        all_AICs[i] = AICs
        true_hyperparams_set[i] = true_hyperparams_arr 
        inferred_hyperparams_set[i] = inferred_hyperparams_collection
    
    results_dict = {'nlls' : all_nlls, 'AICs' : all_AICs, 
                    'true_hyperparams' : true_hyperparams_set,
                    'all_inferred_hyperparams' : inferred_hyperparams_set}
    
    if save_data:
        full_filename = f'{filename}_{true_learning_rule}_{true_model_type}_numtrials_{num_trials}_numoptseeds{num_opt_seeds}_truealpha{fixed_alpha}_truebeta{fixed_beta}_with_two_layer_reg{add_two_layer_reg}.pkl'
        with open(full_filename, 'wb') as file:
            pkl.dump(results_dict, file)

    return results_dict


def plot_single_trial_fit_scores(results_dict, true_learning_rule, true_model_type,
                           savefig=False, filepath='', score_type='nll'):
    assert score_type in ['nll', 'aic']
    assert true_learning_rule in ['gd', 'te']
    assert true_model_type in ['two-layer', 'one-layer']

    score_label = 'Relative NLL'
    if score_type == 'aic':
        score_label = 'Relative AIC'

    if score_type == 'nll':
        A = results_dict['nlls']    # has shape (num_trials, num_models)
    else:
        A = results_dict['AICs']

    num_test_items = A.shape[-1]
    num_trials = A.shape[0]

    # Optional method labels (change these to whatever yours are)
    method_labels = {
        1: '2-layer GD',
        2: '2-layer TE',
        3: '1-layer GD',
        4: '1-layer TE'
    }

    # Prepare data for seaborn
    records = []
    for trial in range(num_trials):
        baseline_val = A[trial, 0]  # method 0
        for method in range(1, 5):
            records.append({
                'Trial': f'Trial {trial+1}',
                'Method': method_labels.get(method, f'Method {method}'),
                score_label: A[trial, method] - baseline_val,
                'Baseline': baseline_val - baseline_val  # This will always be 0
            })

    df = pd.DataFrame(records)

    # Plot: use stripplot instead of boxplot
    plt.figure(figsize=(13, 5))
    sns.stripplot(data=df, x='Trial', y=score_label, hue='Method', dodge=True, size=14, palette='tab10', jitter=False)

    # Overlay black bars for ground truth
    # For each test set, draw a horizontal line at the baseline
    xticks = plt.gca().get_xticks()
    trials = df['Trial'].unique()
    for i, trial in enumerate(trials):
        y = df[df['Trial'] == trial]['Baseline'].iloc[0]
        plt.hlines(y, i - 0.33, i + 0.33, color='black', linewidth=2.5)

    plt.title(f"Single Trial Goodness of Fit", fontsize=20)
    plt.ylabel(score_label)
    plt.xlabel("Trial")
    plt.legend(title='Method')
    plt.tight_layout()

    if savefig:
        plt.savefig(f'{filepath}/single_trial_{score_label}_inference_comparison_ground_truth_{true_model_type}_{true_learning_rule}_numtrials{num_trials}.png')
    else:
        plt.show()



if __name__=="__main__":
    true_learning_rule = 'gd'
    true_model_type = 'one-layer'

    key = jr.PRNGKey(99)
    _, key = jr.split(key, 2)

    true_alpha = 0.003
    true_beta = 10.0
    num_trials = 3


    results_dict = multi_trial_simulated_aic_test(true_learning_rule, true_model_type, num_trials, key, num_opt_seeds=10,
                                                  fixed_alpha=true_alpha, fixed_beta=true_beta, save_data=True, add_two_layer_reg=True, one_layer_scale=0.01)
    

    # with open(f'single_trial_tests_{true_learning_rule}_{true_model_type}_numtrials_{num_trials}_numoptseeds15_truealpha{true_alpha}_truebeta{true_beta}.pkl', 'rb') as file:
    #     results_dict = pkl.load(file)


    filepath = 'single_trial_fit_performance/with_two_layer_reg'
    # with open('single_trial_tests_gd_one-layer_numtrials_1_numoptseeds10_truealpha0.003_truebeta10.0_with_two_layer_regTrue.pkl', 'rb') as file:
    #     results_dict = pkl.load(file)
    
    inferred_hyperparams = results_dict['all_inferred_hyperparams']
    print(inferred_hyperparams.shape)


    trial = 0
    num_hyperparams = [11, 11, 8, 8]
    model_names = ['two-layer_gd', 'two-layer_te', 'one-layer_gd', 'one-layer_te']
    print(results_dict['nlls'][trial])
    print(results_dict['AICs'][trial])
    for i in range(4):
        hyperparams_arr = inferred_hyperparams[trial, i, :num_hyperparams[i]]
        print(f'\n{model_names[i]} inference results:')
        print(f'alpha: {jnp.exp(hyperparams_arr[0])}')
        print(f'beta: {jnp.exp(hyperparams_arr[1])}')
        if i <= 1:
            print(f'w1: {hyperparams_arr[2:5]}')
            print(f'W2: {hyperparams_arr[5:11]}')
        else:
            print(f'W: {hyperparams_arr[2:8]}')

    plot_single_trial_fit_scores(results_dict, true_learning_rule, true_model_type,
                           savefig=True, filepath=filepath, score_type='aic')
    
    # plot_single_trial_fit_scores(results_dict, true_learning_rule, true_model_type,
    #                        savefig=True, filepath=filepath, score_type='nll')
