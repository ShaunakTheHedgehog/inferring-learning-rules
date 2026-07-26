"""K-fold cross-validated single-animal fitting for real mouse data.

Fits one (mouse, model class, CV fold) combination and saves the result, so a SLURM
job array can run every combination in parallel. The fitting objective is the
regularized NLL restricted to the training trials (`masked=True`); the reported test
NLL is the *prior-free* NLL on the held-out trials only.

The weight trajectory is always simulated on the animal's FULL trial sequence -- the
mask only restricts which trials contribute to the likelihood. This is deliberate:
the animal's weights genuinely evolved on every trial it experienced, so masking
trials out of the state-update dynamics would score the model from a state it never
actually occupied.

Usage:
    # explicit (mouse, fold); --model is one of 1L (1-layer linear), 2L (2-layer
    # linear), 2NL (2-layer nonlinear/erf) -- see MODEL_CLASSES below
    python cross_val_fitting.py --model 2NL --mouse_id 13 --fold 0

    # or let a SLURM array index choose the (mouse, fold) pair
    python cross_val_fitting.py --model 2NL --task_id $SLURM_ARRAY_TASK_ID

    # fast end-to-end smoke test (tiny fitting budget)
    python cross_val_fitting.py --model 1L --task_id 0 --dry_run
"""
import argparse
import os
import pickle as pkl
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from globals import MU_ALPHA, MU_BETA, RIDGE_LAMBDA, RIDGE_ALPHABETA
from utils import get_nonlinearity, make_kfold_train_masks, tanh_p
from inference_utils import convert_weights_to_dict, get_num_alphas
from inference_pipeline import (
    run_randomized_optimization_procedure,
    run_train_test_single_learning_forward_trajectory,
)

# model class name -> (inference_model, depth, nln_name), matching globals.MODEL_CLASS_INFO
MODEL_CLASSES = {
    '1L':  ('gd', 'one-layer', 'linear'),
    '2L':  ('gd', 'two-layer', 'linear'),
    '2NL': ('gd', 'two-layer', 'erf'),
}


def load_mouse_data(dataset_name):
    """Load the per-mouse behavioral dict (same file as mouse_inference_data_analysis.load_data)."""
    path = f'{dataset_name}_mouse_data.pkl'
    with open(path, 'rb') as f:
        return pkl.load(f)


def get_mouse_arrays(mouse_dict):
    """Extract (inputs, outputs, rewards) as JAX arrays with the dtypes the pipeline expects.

    `outputs` are stored as floats but index into the model's Q-values, so they must be
    integer-typed (0 = right, 1 = left).
    """
    inputs = jnp.asarray(mouse_dict['inputs'], dtype=jnp.float64)
    outputs = jnp.asarray(mouse_dict['outputs']).astype(jnp.int32)
    rewards = jnp.asarray(mouse_dict['rewards'], dtype=jnp.float64)
    return inputs, outputs, rewards


def check_writable(directory):
    """Fail fast if `directory` is not writable.

    Called before fitting starts: a bad path or an exhausted disk quota should cost a
    second, not a multi-hour job that dies at the final `pkl.dump`.
    """
    probe = os.path.join(directory, f'.writetest_{os.getpid()}')
    try:
        with open(probe, 'wb') as f:
            f.write(b'ok')
        os.remove(probe)
    except OSError as e:
        raise SystemExit(f'ERROR: cannot write results to {directory}: {e}')


def to_numpy(obj):
    """Recursively convert JAX arrays to NumPy inside containers.

    Saved records are then plain NumPy/Python objects, so they load anywhere without
    JAX installed and do not depend on JAX's pickle format staying stable.
    """
    if isinstance(obj, jnp.ndarray):
        return np.asarray(obj)
    if isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(v) for v in obj)
    return obj


def build_task_list(mouse_ids, K):
    """Flat list of (mouse_id, fold) pairs indexed by SLURM array id.

    FOLD-MAJOR order (task_id = fold * num_mice + mouse_index): every mouse's fold 0
    comes first, then every mouse's fold 1, etc. Since SLURM dispatches roughly in index
    order, this yields one train/test fit for *every* mouse as early as possible, with
    later folds completing the cross-validation afterwards. Each fold is also a
    contiguous index range, so it can be submitted alone (e.g. --array=0-34 for fold 0).
    """
    return [(mid, k) for k in range(K) for mid in mouse_ids]


def fit_one_fold(inputs, outputs, rewards, train_mask, depth, nln, inference_model,
                 fit_key, args):
    """Run the randomized multi-seed fit on the training trials, then score the held-out fold."""

    best_w0, min_objective_val, best_hyperparams_arr, optim_dict = run_randomized_optimization_procedure(
        inputs, outputs, rewards, fit_key, depth, nln, inference_model,
        batched=False, masked=True, train_mask=train_mask,
        num_seeds=args.num_seeds, max_iters=args.max_iters, tol=args.tol,
        alpha_init_scale=args.alpha_init_scale, beta_init_scale=args.beta_init_scale,
        w0_init_scale=args.w0_init_scale,
        ridge_lambda=args.ridge_lambda, alphabeta_ridge=args.alphabeta_ridge,
        mu_alpha=args.mu_alpha, mu_beta=args.mu_beta,
        explode_tol=args.explode_tol, burn_in_iters=args.burn_in_iters,
    )

    # `run_randomized_optimization_procedure` only reports a best fit among *converged*
    # seeds, returning None if none converged. Here we instead take the best finite
    # objective over all seeds, so an expensive fit is never discarded; `converged`
    # records whether any seed actually met the gradient tolerance, so fits that relied
    # on a non-converged seed can be filtered or rerun later.
    converged = best_hyperparams_arr is not None
    objective_vals = np.asarray(optim_dict['nlls'], dtype=float)
    if np.any(np.isfinite(objective_vals)):
        best_idx = int(np.nanargmin(np.where(np.isfinite(objective_vals), objective_vals, np.nan)))
        best_hyperparams_arr = optim_dict['hyperparams'][best_idx]
        min_objective_val = objective_vals[best_idx]
        # rebuild best_w0 too, so the record is complete regardless of convergence
        num_alphas = get_num_alphas(inference_model, depth)
        best_w0 = convert_weights_to_dict(best_hyperparams_arr[num_alphas + 1:])
        if not converged:
            print(f'WARNING: no seed converged; falling back to best finite seed '
                f'(idx {best_idx}, objective {min_objective_val:.4f})', flush=True)
    else:
        print('ERROR: no seed produced a finite objective; nothing to score.', flush=True)

    # Score the held-out trials. These NLLs are prior-free (the ridge terms live only in
    # the fitting objective), which is what we want for a predictive-performance measure.
    train_nll = None
    test_nll = None
    if best_hyperparams_arr is not None:
        tr, te, _, _ = run_train_test_single_learning_forward_trajectory(
            best_hyperparams_arr, inputs, outputs, rewards, train_mask, nln, inference_model, None)
        train_nll, test_nll = float(tr), float(te)

    return dict(
        best_w0=best_w0,
        best_hyperparams_arr=best_hyperparams_arr,
        min_objective_val=(None if min_objective_val is None else float(min_objective_val)),
        optim_dict=optim_dict,
        converged=converged,
        train_nll=train_nll,
        test_nll=test_nll,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # what to fit
    p.add_argument('--model', required=True, choices=sorted(MODEL_CLASSES),
                   help='model class to fit')
    p.add_argument('--dataset', default='witten', choices=['witten', 'saxe'])
    p.add_argument('--data_root', default='.',
                   help='directory containing <dataset>_data_GD_analysis_final/')
    p.add_argument('--mouse_id', type=int, default=None, help='mouse id (omit if using --task_id)')
    p.add_argument('--fold', type=int, default=None, help='CV fold index (omit if using --task_id)')
    p.add_argument('--task_id', type=int, default=None,
                   help='SLURM array index; maps to a (mouse, fold) pair via build_task_list')
    p.add_argument('--list_tasks', action='store_true',
                   help='print the number of array tasks (mice x K) and exit')
    p.add_argument("--saturate_inputs", action="store_true",
                           help="apply tanh(5c)/tanh(5) contrast normalization ")

    # cross-validation setup
    p.add_argument('--K', type=int, default=5, help='number of CV folds')
    p.add_argument('--block_length', type=int, default=20, help='trials per contiguous block')
    p.add_argument('--fold_seed', type=int, default=0,
                   help='base PRNG seed for the block->fold assignment (combined with mouse_id). '
                        'MUST be identical across model classes so all models are scored on the '
                        'same folds (paired comparison); change it only for a repeated-K-fold '
                        'replicate with a fresh set of partitions.')

    # optimization
    p.add_argument('--num_seeds', type=int, default=200, help='random restarts per fold')
    p.add_argument('--max_iters', type=int, default=1000, help='max L-BFGS iterations per restart')
    p.add_argument('--tol', type=float, default=1e-3, help='gradient-norm convergence tolerance')
    p.add_argument('--fit_seed', type=int, default=0, help='PRNG seed for optimizer restarts')
    p.add_argument('--alpha_init_scale', type=float, default=1e-2)
    p.add_argument('--beta_init_scale', type=float, default=5e1)
    p.add_argument('--w0_init_scale', type=float, default=1e-1)
    p.add_argument('--explode_tol', type=float, default=1e7)
    p.add_argument('--burn_in_iters', type=int, default=1_000)

    # regularization (defaults from globals.py)
    p.add_argument('--ridge_lambda', type=float, default=RIDGE_LAMBDA)
    p.add_argument('--alphabeta_ridge', type=float, default=RIDGE_ALPHABETA)
    p.add_argument('--mu_alpha', type=float, default=MU_ALPHA)
    p.add_argument('--mu_beta', type=float, default=MU_BETA)

    # output
    p.add_argument('--out_dir', default='cross_val_results', help='root directory for results')
    p.add_argument('--overwrite', action='store_true',
                   help='refit even if the result file already exists')
    p.add_argument('--dry_run', action='store_true',
                   help='tiny fitting budget for a fast end-to-end smoke test')
    args = p.parse_args()

    all_data = load_mouse_data(args.dataset)
    mouse_ids = sorted(all_data.keys())
    tasks = build_task_list(mouse_ids, args.K)

    if args.list_tasks:
        n = len(mouse_ids)
        print(f'{n} mice x K={args.K} folds = {len(tasks)} array tasks '
              f'(use --array=0-{len(tasks)-1})')
        print('fold-major ordering, so each fold is a contiguous index range:')
        for k in range(args.K):
            print(f'  fold {k}: --array={k*n}-{(k+1)*n-1}')
        return

    # resolve which (mouse, fold) this invocation handles
    if args.task_id is not None:
        if not (0 <= args.task_id < len(tasks)):
            raise SystemExit(f'--task_id {args.task_id} out of range 0..{len(tasks)-1}')
        mouse_id, fold = tasks[args.task_id]
    else:
        if args.mouse_id is None or args.fold is None:
            raise SystemExit('provide either --task_id, or both --mouse_id and --fold')
        mouse_id, fold = args.mouse_id, args.fold
    if mouse_id not in all_data:
        raise SystemExit(f'mouse {mouse_id} not in {args.dataset} data; available: {mouse_ids}')
    if not (0 <= fold < args.K):
        raise SystemExit(f'--fold {fold} out of range 0..{args.K-1}')

    if args.dry_run:
        args.num_seeds, args.max_iters = 2, 20

    inference_model, depth, nln_name = MODEL_CLASSES[args.model]
    nln = get_nonlinearity(nln_name)

    # fold_seed is part of the filename: a different seed is a different partition, so
    # results from separate seeds (repeated K-fold) must not silently collide.
    base_folder = args.out_dir
    if args.saturate_inputs:
        base_folder = f'{args.out_dir}_sat_inputs'
        print(f'Will save to folder: {base_folder}', flush=True)

    out_dir = os.path.join(base_folder, args.model, f'mouse_{mouse_id}')
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise SystemExit(f'ERROR: cannot create output directory {out_dir}: {e}')
    check_writable(out_dir)  # fail before fitting, not after a multi-hour run
    out_path = os.path.join(out_dir, f'fold_{fold}_foldseed_{args.fold_seed}.pkl')
    if os.path.exists(out_path) and not args.overwrite:
        print(f'{out_path} already exists; use --overwrite to refit. Exiting.', flush=True)
        return

    inputs, outputs, rewards = get_mouse_arrays(all_data[mouse_id])
    T = int(outputs.shape[0])

    # apply front-end saturating nonlinearity if specified
    if args.saturate_inputs:
        print(f'Using saturating nonlinearity on inputs', flush=True)
        inputs = tanh_p(inputs)

    # The fold key is derived from (fold_seed, mouse_id), so each mouse gets its own
    # partition draw while every task for that mouse -- all K folds, all model classes --
    # regenerates exactly the same partition without needing to coordinate. That identity
    # is what makes the folds disjoint/exhaustive and the model comparison paired.
    # Bumping --fold_seed gives a fresh set of partitions for a repeated-K-fold replicate.
    fold_key = jr.fold_in(jr.PRNGKey(args.fold_seed), mouse_id)
    train_masks = make_kfold_train_masks(T, fold_key, K=args.K, block_length=args.block_length)
    train_mask = train_masks[fold]
    n_train, n_test = int(jnp.sum(train_mask)), int(jnp.sum(~train_mask))

    print(f'=== {args.dataset} mouse {mouse_id} | model {args.model} '
          f'({depth}, {nln_name}) | fold {fold}/{args.K} ===', flush=True)
    print(f'T={T} trials | train={n_train} test={n_test} '
          f'(held-out fraction {n_test/T:.3f})', flush=True)

    t0 = time.time()
    result = fit_one_fold(inputs, outputs, rewards, train_mask, depth, nln, inference_model,
                          jr.PRNGKey(args.fit_seed), args)
    elapsed = time.time() - t0

    if result['test_nll'] is not None:
        print(f"\ntrain NLL={result['train_nll']:.3f} | TEST NLL={result['test_nll']:.3f} "
              f"| test NLL/trial={result['test_nll']/n_test:.5f} "
              f"(chance={np.log(2):.5f})", flush=True)
    print(f'converged={result["converged"]} '
          f'elapsed={elapsed:.1f}s', flush=True)

    record = dict(
        # identity
        dataset=args.dataset, mouse_id=mouse_id, model=args.model,
        inference_model=inference_model, depth=depth, nln_name=nln_name, saturate_inputs=args.saturate_inputs,
        # fold metadata (enough to reproduce the split exactly)
        fold=fold, K=args.K, block_length=args.block_length, fold_seed=args.fold_seed,
        train_mask=np.asarray(train_mask), num_trials=T, n_train=n_train, n_test=n_test,
        # fit settings
        num_seeds=args.num_seeds, max_iters=args.max_iters, tol=args.tol, fit_seed=args.fit_seed,
        ridge_lambda=args.ridge_lambda, alphabeta_ridge=args.alphabeta_ridge,
        mu_alpha=args.mu_alpha, mu_beta=args.mu_beta,
        elapsed_sec=elapsed,
        **result,
    )
    record = to_numpy(record)  # drop JAX arrays -> loads anywhere, independent of JAX version

    # write to a temp file then rename, so a job killed mid-write (e.g. by a SLURM time
    # limit) cannot leave a half-written pickle for a later aggregation step to choke on
    tmp_path = f'{out_path}.tmp{os.getpid()}'
    with open(tmp_path, 'wb') as f:
        pkl.dump(record, f)
    os.replace(tmp_path, out_path)
    print(f'saved -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
