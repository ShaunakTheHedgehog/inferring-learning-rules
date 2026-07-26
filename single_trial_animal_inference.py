import os
os.environ['OMP_NUM_THREADS'] = str(1)  # match SLURM cpus-per-task
os.environ['MKL_NUM_THREADS'] = str(1)

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
import numpy as np
import pickle as pkl
import argparse

import matplotlib.pyplot as plt

from utils import get_nonlinearity, calculate_simple_mouse_ma_bias, compute_rightward_statistics, tanh_p
from inference_utils import get_num_alphas
from inference_pipeline import run_randomized_optimization_procedure
from globals import *


def single_trial_mouse_data_validation(dataset_name, mouse_behav_dict, inference_model, depth, nln, key, num_seeds=200, max_iters=2_000, tol=1e-3,
                                       fixed_alpha_W1=None, alpha_init_scale=1e-2, beta_init_scale=5e0, w0_init_scale=1e-1, 
                                       ridge_lambda=0.5, alphabeta_ridge=0.5, mu_alpha=MU_ALPHA, mu_beta=MU_BETA,
                                       explode_tol=1e7, burn_in_iters=1_000, saturate=False):
    assert inference_model in ['gd', 'full_metric_gd', 'pathway_metric_gd', 'fixed_metric_gd']
    assert depth in ['two-layer', 'one-layer']
    assert dataset_name in ['witten', 'saxe']
    
    _, key, skey = jr.split(key, 3)

    num_alphas = get_num_alphas(inference_model, depth)
    num_weights = DEPTH_TO_NUM_WEIGHTS[depth]
    total_num_hyperparams = num_alphas + num_weights + NUM_BETAS

    print(f'FITTING TO REAL BEHAVIORAL DATA WITH A {depth} NETWORK USING {inference_model} INFERENCE...', flush=True)

    p = total_num_hyperparams
    inferred_hyperparams_collection = np.zeros(total_num_hyperparams)

    inputs = mouse_behav_dict['inputs']
    if saturate:
        inputs = tanh_p(inputs)
    
    outputs = mouse_behav_dict[DATASET_TO_CHOICE_KEYWORD[dataset_name]].astype(jnp.int32)
        
    rewards = mouse_behav_dict['rewards'] 

    inferred_w0, min_nll, inferred_hyperparams, optim_dict = run_randomized_optimization_procedure(inputs, outputs, rewards, skey, depth, nln, inference_model, batched=False,
                                                alpha_W1=fixed_alpha_W1, num_seeds=num_seeds, max_iters=max_iters, tol=tol,
                                                alpha_init_scale=alpha_init_scale, beta_init_scale=beta_init_scale, w0_init_scale=w0_init_scale,
                                                ridge_lambda=ridge_lambda, alphabeta_ridge=alphabeta_ridge, mu_alpha=mu_alpha, mu_beta=mu_beta,
                                                explode_tol=explode_tol, burn_in_iters=burn_in_iters)
    
    nll = min_nll
    inferred_hyperparams_collection = inferred_hyperparams
    print(f'Minimum inferred (training) NLL: {min_nll}', flush=True)

    AIC = 2 * nll + 2 * p

    print(f'\nNLL: {nll}', flush=True)
    print(f'AIC: {AIC}', flush=True)
    print(f'Inferred hyperparameters: {inferred_hyperparams_collection}', flush=True)
    
    return nll, AIC, inferred_hyperparams_collection, inferred_w0, optim_dict


if __name__ == "__main__":
    
    # read in configuration parameters from SLURM script
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouse_idx', type=int, default=0)
    parser.add_argument('--nln_type', type=str, default='linear')
    parser.add_argument('--depth', type=str, default='two-layer')
    parser.add_argument('--inference_model', type=str, default='gd')
    parser.add_argument("--saturate_inputs", action="store_true",
                       help="apply tanh(5c)/tanh(5) contrast normalization ")
    
    parser.add_argument('--ridge_lambda', type=float, default=0.5)
    parser.add_argument('--alphabeta_ridge', type=float, default=0.5)
    
    parser.add_argument('--num_opt_seeds', type=int, default=200)
    parser.add_argument('--max_iters', type=int, default=2_000)

    parser.add_argument('--dataset', type=str, default='witten')
    parser.add_argument('--filepath', type=str)
    
    args = parser.parse_args()
    mouse_idx = args.mouse_idx
    
    nln_name = args.nln_type
    nln = get_nonlinearity(nln_name)
    
    depth = args.depth
    inference_model = args.inference_model
    ridge_lambda = args.ridge_lambda 
    alphabeta_ridge = args.alphabeta_ridge
    
    num_opt_seeds = args.num_opt_seeds
    max_iters = args.max_iters
    
    filepath = args.filepath
    dataset_name = args.dataset
    assert dataset_name in ['witten', 'saxe']
    
    # if specified file path doesn't currently exist, create it
    os.makedirs(filepath, exist_ok=True)
    
    key = jr.PRNGKey(2026 + mouse_idx)    # ensure unique key for each mouse
    
    all_mice_dict = pkl.load(open(f'{dataset_name}_mouse_data.pkl', 'rb'))
    choice_keyword = DATASET_TO_CHOICE_KEYWORD[dataset_name]
   
    mouse_ids = list(all_mice_dict.keys())
    
    mouse_id = mouse_ids[mouse_idx]
    print(f'Mouse {mouse_id}', flush=True)
    
    mouse_behav_dict = all_mice_dict[mouse_id]
    
    inputs = mouse_behav_dict['inputs']
    outputs = mouse_behav_dict[choice_keyword].astype(jnp.int32)
    rewards = mouse_behav_dict['rewards']

    rightward_stats = compute_rightward_statistics(inputs, outputs)
    ma_bias = calculate_simple_mouse_ma_bias(outputs)
    print(ma_bias, flush=True)
    
    session_accs = mouse_behav_dict['session_accs']
    print(session_accs, flush=True)

    try:
        nll, AIC, inferred_hyperparams, inferred_w0, optim_dict = single_trial_mouse_data_validation(dataset_name, mouse_behav_dict, inference_model, depth, nln, key, 
                                                                            num_seeds=num_opt_seeds, max_iters=max_iters, ridge_lambda=ridge_lambda, alphabeta_ridge=alphabeta_ridge,
                                                                            mu_alpha=MU_ALPHA, mu_beta=MU_BETA, explode_tol=1e7, burn_in_iters=1_000, saturate=args.saturate_inputs)
    except Exception as e:
        print(f"Error processing mouse {mouse_id}: {e}", flush=True)
        nll, AIC, inferred_hyperparams, optim_dict = None, None, None, None

    print(f'Inferred alpha: {jnp.exp(inferred_hyperparams[0])}', flush=True)
    print(f'Inferred beta: {jnp.exp(inferred_hyperparams[1])}', flush=True)
    print(f'Lowest NLL: {nll}', flush=True)
    
    mouse_results = {'mouse_id' : mouse_id, 'inputs' : inputs, 'outputs' : outputs, 'rewards' : rewards,
                     'ma_bias' : ma_bias, 'session_accs' : session_accs, 'rightward_stats' : rightward_stats,
                     'nll' : nll, 'AIC' : AIC, 'inferred_hyperparams' : inferred_hyperparams, 'optim_dict' : optim_dict,
                     'saturate_inputs': args.saturate_inputs}

    with open(f'{filepath}/mouse_{mouse_id}_results_{depth}_{inference_model}_{nln_name}.pkl', 'wb') as f:
        pkl.dump(mouse_results, f)

