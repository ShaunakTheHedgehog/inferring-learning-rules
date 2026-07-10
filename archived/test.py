# --- path shim: allow running this legacy script standalone after the move to legacy/ ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path shim ---
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax.tree_util import tree_map
import optax
import optax.tree_utils as otu
from functools import partial
import numpy as np
import pickle as pkl
import os

from globals import *
from utils import get_nonlinearity, calculate_simple_mouse_ma_bias
from inference_pipeline import run_single_learning_forward_trajectory, compute_nll, compute_hessian


def find_best_hyperparams(optim_dict):
    all_nlls = jnp.asarray(optim_dict['nlls'])
    bad_run_idxs = jnp.asarray(optim_dict['bad_run_idxs']).astype(int)

    if len(bad_run_idxs) > 0:
        bad_nlls = all_nlls[bad_run_idxs]
        if jnp.min(bad_nlls) == jnp.min(all_nlls):
            print('NOTE: HERE WE ARE USING A NON-CONVERGED HYPERPARAM VECTOR!', flush=True)
        
    all_hyperparams = optim_dict['hyperparams']
    best_idx = jnp.argmin(all_nlls)
    
    return all_hyperparams[best_idx]



def run_convergence_diagnostics(dataset_name):
    assert dataset_name in ['witten', 'saxe'], f'Invalid dataset name: {dataset_name}'

    mouse_behav_data = pkl.load(open(f'{dataset_name}_data_GD_analysis_OLDER/{dataset_name}_mouse_data.pkl', 'rb'))
    ids = list(mouse_behav_data.keys())

    bads = []

    for key in MODEL_CLASS_INFO.keys():
        inference_model, depth, nln_name, num_params, _ = MODEL_CLASS_INFO[key]
        nln = get_nonlinearity(nln_name)

        dir = f'hessian_test/{dataset_name}_data_GD_fits_final/{inference_model}_{nln_name}_{depth}'
        for id in ids:
            file = f'{dir}/mouse_{id}_results_{depth}_{inference_model}_{nln_name}.pkl'

            # print(f'Testing Mouse {id}:')
            try:
                results_dict = pkl.load(open(file, 'rb'))
            except FileNotFoundError:
                print(f'File not found for Mouse {id}: {file}')
                continue
            x0 = results_dict['inferred_hyperparams']
            inputs = results_dict['inputs']
            outputs = results_dict['outputs']
            rewards = results_dict['rewards']

            nll = compute_nll(x0, inputs, outputs, rewards, nln, inference_model, False, alpha_W1=None, num_weights=num_params, ridge_lambda=RIDGE_LAMBDA,
                    alphabeta_ridge=RIDGE_ALPHABETA, mu_alpha=MU_ALPHA, mu_beta=MU_BETA)
            # print(f'NLL: {nll}')

            optim_dict = results_dict['optim_dict']
            best_x0 = find_best_hyperparams(optim_dict)
            bad_run_idxs = optim_dict['bad_run_idxs']
            nlls = optim_dict['nlls']
            if len(bad_run_idxs) > 0:
                bad_run_idxs = jnp.asarray(bad_run_idxs)
                bad_nlls = nlls[bad_run_idxs]
                # print(f'Min BAD NLL: {jnp.min(bad_nlls)}', flush=True)

                if jnp.min(bad_nlls) < nll:
                    print(f'OPTIMIZATION ERROR. BAD SMALLER NLL FOUND for Mouse {id} and {key} model.', flush=True)
                else:
                    assert best_x0.all() == x0.all()
            else:
                assert best_x0.all() == x0.all()

            grad_nll = grad(compute_nll)(x0, inputs, outputs, rewards, nln, inference_model, False, alpha_W1=None, num_weights=num_params, ridge_lambda=RIDGE_LAMBDA,
                    alphabeta_ridge=RIDGE_ALPHABETA, mu_alpha=MU_ALPHA, mu_beta=MU_BETA)
            # print(f'Grad NLL: {grad_nll}')
            # print(f'Grad NLL Norm: {jnp.linalg.norm(grad_nll)}')

            _, eigvals = compute_hessian(x0, inputs, outputs, rewards, nln, inference_model, False, alpha_W1=None, num_weights=num_params, ridge_lambda=RIDGE_LAMBDA,
                    alphabeta_ridge=RIDGE_ALPHABETA, mu_alpha=MU_ALPHA, mu_beta=MU_BETA)
            # print(f'Hessian Eigvals: {eigvals}')
            # print('-----------------------------')
            # print('\n')

            if jnp.any(eigvals <= 0) or jnp.linalg.norm(grad_nll) > 1e-3:
                print(f'Bad fit for Mouse {id} in {key} model:')

    print('Done!')        



dataset_name = 'witten'
run_convergence_diagnostics(dataset_name)
print(1./0)
mouse_behav_data = pkl.load(open(f'{dataset_name}_data_GD_analysis_OLDER/{dataset_name}_mouse_data.pkl', 'rb'))
ids = list(mouse_behav_data.keys())

bads = []

for key in MODEL_CLASS_INFO.keys():
    inference_model, depth, nln_name, num_params, _ = MODEL_CLASS_INFO[key]
    nln = get_nonlinearity(nln_name)

    dir = f'hessian_test/{dataset_name}_data_GD_fits_final/{inference_model}_{nln_name}_{depth}'
    for id in ids:
        file = f'{dir}/mouse_{id}_results_{depth}_{inference_model}_{nln_name}.pkl'

        # print(f'Testing Mouse {id}:')
        try:
            results_dict = pkl.load(open(file, 'rb'))
        except FileNotFoundError:
            print(f'File not found for Mouse {id}: {file}')
            continue
        x0 = results_dict['inferred_hyperparams']
        inputs = results_dict['inputs']
        outputs = results_dict['outputs']
        rewards = results_dict['rewards']

        ma_bias = calculate_simple_mouse_ma_bias(outputs)

