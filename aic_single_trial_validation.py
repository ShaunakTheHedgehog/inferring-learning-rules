import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax import debug
from jax.tree_util import tree_map
import numpy as np
import pickle as pkl


import concurrent.futures
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
                                    num_opt_seeds=15, fixed_alpha=None, fixed_beta=None, fixed_w0=None):
        
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
                              num_train_sets=1, num_test_sets=0, vary_inputs=False, true_w0=fixed_w0)
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

        print(f'\nPerforming inference in a {model_type} network with the {test_learning_rule} learning rule...')
        num_alpha_blocks = 1
        _, min_nll, hyperparams_arr = run_randomized_optimization_procedure(inputs, outputs, rewards, curr_key, learning_rule=test_learning_rule, model_type=model_type,
                                                                            num_seeds=num_opt_seeds, max_iters=1_000, 
                                                                            tol=1e-6, method='optax',
                                                                            alpha_init_scale=0.1, beta_init_scale=1., w0_init_scale=0.1, 
                                                                            num_alpha_blocks=num_alpha_blocks, l2_lambda=0., batched=False)
        nlls[j + 1] = min_nll
        inferred_hyperparams_collection[j, :len(hyperparams_arr)] = hyperparams_arr
        print(f'Inferred (training) NLL with {test_learning_rule} as the true learning rule: {min_nll}')

    AICs = 2 * nlls + 2 * ps

    return nlls, AICs, true_hyperparams_arr, inferred_hyperparams_collection




# Helper function for parallel execution (must be top-level for pickling)
def _single_trial_worker(args):
    (true_learning_rule, true_model_type, key, session_info, num_opt_seeds, fixed_alpha, fixed_beta, fixed_w0) = args
    return single_trial_simulated_aic_test(
        true_learning_rule, true_model_type, key, session_info=session_info,
        num_opt_seeds=num_opt_seeds, fixed_alpha=fixed_alpha,
        fixed_beta=fixed_beta, fixed_w0=fixed_w0
    )

def multi_trial_simulated_aic_test(true_learning_rule, true_model_type, num_trials, key, session_info=(40, 4, 30, 10), 
                                   num_opt_seeds=15, fixed_alpha=None, fixed_beta=None, fixed_w0=None, save_data=True, filename='single_trial_tests'):
    assert true_learning_rule in ['gd', 'te']
    assert true_model_type in ['two-layer', 'one-layer']
    true_num_params = 11
    if true_model_type == 'one-layer':
        true_num_params = 8

    all_nlls = np.zeros((num_trials, 5))
    all_AICs = np.zeros((num_trials, 5))
    true_hyperparams_set = np.zeros((num_trials, true_num_params))
    inferred_hyperparams_set = np.zeros((num_trials, 4, 11))

    # Prepare keys for each trial
    keys = []
    curr_key = key
    for _ in range(num_trials):
        _, curr_key = jr.split(curr_key, 2)
        keys.append(curr_key)

    # Prepare arguments for each trial
    args_list = [
        (true_learning_rule, true_model_type, keys[i], session_info, num_opt_seeds, fixed_alpha, fixed_beta, fixed_w0)
        for i in range(num_trials)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_single_trial_worker, args_list))

    for i, (nlls, AICs, true_hyperparams_arr, inferred_hyperparams_collection) in enumerate(results):
        all_nlls[i] = nlls 
        all_AICs[i] = AICs
        true_hyperparams_set[i] = true_hyperparams_arr 
        inferred_hyperparams_set[i] = inferred_hyperparams_collection

    results_dict = {'nlls' : all_nlls, 'AICs' : all_AICs, 
                    'true_hyperparams' : true_hyperparams_set,
                    'all_inferred_hyperparams' : inferred_hyperparams_set}
    
    if save_data:
        full_filename = f'{filename}_{true_learning_rule}_{true_model_type}_numtrials_{num_trials}_numoptseeds{num_opt_seeds}_truealpha{fixed_alpha}_truebeta{fixed_beta}.pkl'
        with open(full_filename, 'wb') as file:
            pkl.dump(results_dict, file)

    return results_dict


if __name__=="__main__":
    true_learning_rule = 'te'
    true_model_type = 'two-layer'

    key = jr.PRNGKey(98)
    _, key = jr.split(key, 2)

    true_alpha = 0.005
    true_beta = 10.


    nlls, AICs, true_hyperparams_arr, inferred_hyperparams_collection = single_trial_simulated_aic_test(true_learning_rule, true_model_type, key, num_opt_seeds=15,
                                                                                                        fixed_alpha=true_alpha, fixed_beta=true_beta)
    
    print(f'NLLs: {nlls}')
    print(f'AICs: {AICs}')

    print(true_hyperparams_arr)
    print(inferred_hyperparams_collection)
