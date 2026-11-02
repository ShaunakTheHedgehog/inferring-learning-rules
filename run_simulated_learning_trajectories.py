import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pickle as pkl

import os 
import argparse

from utils import get_nonlinearity
from inference_utils import extract_weights_from_hyperparams, get_num_alphas, get_depth, train_gd_network
from inference_pipeline import compute_hessian
from globals import *


def extract_fitted_params(mouse_results_dict, inference_model, return_dict=False):
    hyperparams, _ = find_best_hyperparams(mouse_results_dict['optim_dict'])
    w0 = extract_weights_from_hyperparams(hyperparams, inference_model, return_dict=return_dict)
    depth = get_depth(hyperparams, inference_model)
    num_alphas = get_num_alphas(inference_model, depth)
    log_alphas = hyperparams[:num_alphas]
    lrs = jnp.exp(log_alphas)
    beta = jnp.exp(hyperparams[num_alphas])
    return w0, depth, lrs, beta
    
    
def extract_lrs_beta_and_w0(hyperparams, inference_model, return_dict=False):
    w0 = extract_weights_from_hyperparams(hyperparams, inference_model, return_dict=return_dict)
    depth = get_depth(hyperparams, inference_model)
    num_alphas = get_num_alphas(inference_model, depth)
    log_alphas = hyperparams[:num_alphas]
    lrs = jnp.exp(log_alphas)
    beta = jnp.exp(hyperparams[num_alphas])
    return lrs, beta, w0
    

def find_best_hyperparams(optim_dict):
    convergence_failed=False
    all_nlls = jnp.asarray(optim_dict['nlls'])
    bad_run_idxs = jnp.asarray(optim_dict['bad_run_idxs']).astype(int)

    if len(bad_run_idxs) > 0:
        bad_nlls = all_nlls[bad_run_idxs]
        if jnp.min(bad_nlls) == jnp.min(all_nlls):
            print('NOTE: HERE WE ARE USING A NON-CONVERGED HYPERPARAM VECTOR!', flush=True)
            convergence_failed=True
        
    all_hyperparams = optim_dict['hyperparams']
    best_idx = jnp.argmin(all_nlls)
    
    return all_hyperparams[best_idx], convergence_failed
    


def get_resampled_model_outputs(mouse_results_dict, inference_model, nln_name, filepath, key, 
                                num_test_sets=1_000, MAP_sampling=False):
    
    nln = get_nonlinearity(nln_name)
    inputs = mouse_results_dict['inputs']
    outputs = mouse_results_dict['outputs'].astype(jnp.int32)
    rewards = mouse_results_dict['rewards'] 
    targets = 1 - ((outputs + rewards.astype(jnp.int32)) % 2)
    targets = targets.reshape((-1, 1))
    
    optim_dict = mouse_results_dict['optim_dict']
    
    # best_converged_hyperparams = mouse_results_dict['inferred_hyperparams']
    # find lowest NLL hyperparams from optimization dictionary
    inferred_hyperparams, _ = find_best_hyperparams(optim_dict)

    all_model_outputs = []

    _, depth, _, _ = extract_fitted_params(mouse_results_dict, inference_model, return_dict=True)

    print(f'Depth is {depth}', flush=True)

    num_weights = NUM_TWO_LAYER_WEIGHTS
    if depth == 'one-layer':
        num_weights = NUM_ONE_LAYER_WEIGHTS

    skey, key = jr.split(key)
    
    # sample hyperparam vectors from the Laplace approximation to the posterior 
    x0 = inferred_hyperparams

    xs = None
    sampling_tag = ''

    if MAP_sampling:
        H, _ = compute_hessian(inferred_hyperparams, inputs, outputs, rewards, nln, inference_model, batched=False, alpha_W1=None, num_weights=num_weights, 
                            ridge_lambda=RIDGE_LAMBDA, alphabeta_ridge=RIDGE_ALPHABETA, mu_alpha=MU_ALPHA, mu_beta=MU_BETA)
        Sigma0 = jnp.linalg.inv(H)
        xs = jr.multivariate_normal(skey, x0, Sigma0, (num_test_sets,))
        sampling_tag = '_MAP_Laplace'
    
    # if Hessian has any eigenvalues that are negative or that blow up (for whatever reason), avoid MAP sampling
    if (not MAP_sampling) or (not jnp.isfinite(xs).all()):
        print(f"Proceeding without MAP sampling", flush=True)
        xs = jnp.tile(x0, (num_test_sets, 1))
    
    print('Starting training process...', flush=True)

    for i in range(num_test_sets):
        key, subkey = jr.split(key)

        curr_hyperparams = xs[i]
        curr_lrs, curr_beta, curr_w0 = extract_lrs_beta_and_w0(curr_hyperparams, inference_model, return_dict=True)

        _, _, results_dict = train_gd_network(nln_name, curr_w0, inputs, targets, curr_lrs[0], curr_beta,
                                              subkey, num_grading_levels=NUM_GRADING_LEVELS, print_stuff=False)

        model_outputs = results_dict['outputs']
        model_outputs = jnp.asarray(model_outputs)
        all_model_outputs.append(model_outputs)


    # make all_model_outputs a numpy array of dtype bool
    all_model_outputs = np.array(all_model_outputs).astype(bool) 
    if filepath is not None:
        np.save(f'{filepath}/all_model_outputs{sampling_tag}.npy', all_model_outputs)
        
        xs = np.asarray(xs)
        np.save(f'{filepath}/all_model_init_hyperparams{sampling_tag}.npy', xs)

    return all_model_outputs


if __name__ == "__main__":

    dataset_to_key = {'witten': 1, 'saxe': 2}

    parser = argparse.ArgumentParser()
    parser.add_argument('--mouse_idx', type=int, default=0)
    parser.add_argument('--nln_type', type=str, default='linear')
    parser.add_argument('--depth', type=str, default='two-layer')
    parser.add_argument('--inference_model', type=str, default='gd')
    parser.add_argument('--num_test_sets', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='witten')
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--mouse_ids', nargs='*', default=None, help='List of mouse IDs to run experiments on. If not specified, uses default list.')
    parser.add_argument('--MAP_sampling', action='store_true', help='Sample hyperparams from the Laplace approximation to the posterior; otherwise simulate the MAP hyperparams only.')
    args = parser.parse_args()
    
    mouse_idx = args.mouse_idx
    
    nln_name = args.nln_type

    depth = args.depth
    inference_model = args.inference_model
    
    num_test_sets = args.num_test_sets
    
    using_MAP_sampling = args.MAP_sampling

    dataset_name = args.dataset
    
    filepath = args.filepath
    
    key = jr.PRNGKey(1000 * dataset_to_key[dataset_name] + mouse_idx)    # ensure unique key for each mouse
    key, _ = jr.split(key)
    
    mouse_ids = args.mouse_ids if args.mouse_ids is not None else ALL_WITTEN_LEARNERS
    
    mouse_id = mouse_ids[mouse_idx]
    print(f'Mouse {mouse_id}', flush=True)
    
    full_filepath = f'{filepath}/{inference_model}_{nln_name}_{depth}_results/mouse_{mouse_id}'
    os.makedirs(full_filepath, exist_ok=True)
    
    try:
        with open(f'{dataset_name}_fits_CLEANED/{inference_model}_{nln_name}_{depth}/mouse_{mouse_id}_results_{depth}_{inference_model}_{nln_name}.pkl', 'rb') as f:
            results_dict = pkl.load(f)
    
        MAP_tag = '_MAP_Laplace' if using_MAP_sampling else ''
        candidate_filename = f'{full_filepath}/all_model_outputs{MAP_tag}.npy'
        path_exists = os.path.exists(candidate_filename)
    
        if not path_exists:
            print(f'Running routine for {depth} {nln_name} {inference_model} networks...', flush=True)
            all_model_outputs = get_resampled_model_outputs(results_dict, inference_model, nln_name, full_filepath, key, num_test_sets, MAP_sampling=using_MAP_sampling)
            print(all_model_outputs.shape, flush=True)
        else:
            print('File already exists -- no need to run anything here!', flush=True)

        print(f'Data collection for Mouse {mouse_id} complete!', flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        raise