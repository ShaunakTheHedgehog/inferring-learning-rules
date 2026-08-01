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
from utils import (get_nonlinearity, make_kfold_train_masks, tanh_p, flip_outputs,
                   check_writable, to_numpy)
from inference_utils import convert_weights_to_dict, get_num_alphas
from inference_pipeline import (
    run_randomized_optimization_procedure,
    run_train_test_single_learning_forward_trajectory,
)
from reinforce_pipeline import (
    NUM_WEIGHTS as RF_NUM_WEIGHTS,
    REINFORCE_MODELS,
    run_RF_randomized_optimization_procedure,
    run_train_test_single_RF_forward_trajectory,
)

# model class name -> (inference_model, depth, nln_name), matching globals.MODEL_CLASS_INFO
MODEL_CLASSES = {
    '1L':  ('gd', 'one-layer', 'linear'),
    '2L':  ('gd', 'two-layer', 'linear'),
    '2NL': ('gd', 'two-layer', 'erf'),
}

# Both model families are cross-validated by this script, on IDENTICAL folds: the fold
# partition depends only on (fold_seed, mouse_id), never on the model, which is what
# makes NN-vs-RF comparison paired. The families differ only in which optimizer runs,
# which penalty terms apply, and the choice coding (see the flip in main()).
ALL_MODELS = sorted(MODEL_CLASSES) + sorted(REINFORCE_MODELS)


def model_family(model):
    """'RF' for the REINFORCE models, 'NN' for the gradient-descent network models."""
    return 'RF' if model in REINFORCE_MODELS else 'NN'


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


def build_task_list(mouse_ids, K):
    """Flat list of (mouse_id, fold) pairs indexed by SLURM array id.

    FOLD-MAJOR order (task_id = fold * num_mice + mouse_index): every mouse's fold 0
    comes first, then every mouse's fold 1, etc. Since SLURM dispatches roughly in index
    order, this yields one train/test fit for *every* mouse as early as possible, with
    later folds completing the cross-validation afterwards. Each fold is also a
    contiguous index range, so it can be submitted alone (e.g. --array=0-34 for fold 0).
    """
    return [(mid, k) for k in range(K) for mid in mouse_ids]


def fit_one_fold(inputs, outputs, rewards, train_mask, model, depth, nln, inference_model,
                 fit_key, args):
    """Run the randomized multi-seed fit on the training trials, then score the held-out fold."""
    is_rf = model_family(model) == 'RF'

    if is_rf:
        best_w0, min_objective_val, best_hyperparams_arr, optim_dict = run_RF_randomized_optimization_procedure(
            inputs, outputs, rewards, fit_key, model,
            masked=True, train_mask=train_mask,
            num_seeds=args.num_seeds, max_iters=args.max_iters, tol=args.tol,
            alpha_init_scale=args.alpha_init_scale, c_init_scale=args.c_init_scale,
            w0_init_scale=args.w0_init_scale,
            ridge_lambda=args.ridge_lambda, ridge_alpha=args.ridge_alpha, ridge_c=args.ridge_c,
            mu_alpha=args.mu_alpha, mu_c=args.mu_c,
            explode_tol=args.explode_tol, burn_in_iters=args.burn_in_iters,
        )
    else:
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
        # rebuild best_w0 too, so the record is complete regardless of convergence.
        # RF weights are a flat (3,) vector; the NN weights are a pytree.
        if is_rf:
            best_w0 = best_hyperparams_arr[-RF_NUM_WEIGHTS:]
        else:
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
        if is_rf:
            tr, te, _, _ = run_train_test_single_RF_forward_trajectory(
                best_hyperparams_arr, model, inputs, outputs, rewards, train_mask)
        else:
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
    p.add_argument('--model', required=True, choices=ALL_MODELS,
                   help='model class to fit: 1L/2L/2NL (GD networks) or RF_*/ (REINFORCE)')
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
    p.add_argument('--max_iters', type=int, default=None,
                   help='max L-BFGS iterations per restart (default: 1000 for NN, 2000 for RF)')
    p.add_argument('--tol', type=float, default=1e-3, help='gradient-norm convergence tolerance')
    p.add_argument('--fit_seed', type=int, default=0, help='PRNG seed for optimizer restarts')
    p.add_argument('--alpha_init_scale', type=float, default=1e-2)
    p.add_argument('--w0_init_scale', type=float, default=1e-1)
    p.add_argument('--explode_tol', type=float, default=1e7)
    p.add_argument('--burn_in_iters', type=int, default=1_000)
    p.add_argument('--beta_init_scale', type=float, default=5e1, help='NN only')
    # RF only. c = alpha*beta is O(alpha), so an init scale of order 1 badly inflates the
    # fitted intercept; 1e-4 keeps every restart near the plain-REINFORCE point c = 0.
    p.add_argument('--c_init_scale', type=float, default=1e-4, help='RF only')

    # Regularization. The families take different penalty terms, so the two shared flags
    # resolve to family-appropriate defaults below: NN uses the globals.py priors, RF is
    # fit unregularized (pure MLE).
    p.add_argument('--ridge_lambda', type=float, default=None,
                   help=f'w0 ridge (default: {RIDGE_LAMBDA} for NN, 0 for RF)')
    p.add_argument('--mu_alpha', type=float, default=None,
                   help=f'log-alpha prior mean (default: {MU_ALPHA:.4f} for NN, 0 for RF)')
    p.add_argument('--alphabeta_ridge', type=float, default=RIDGE_ALPHABETA, help='NN only')
    p.add_argument('--mu_beta', type=float, default=MU_BETA, help='NN only')
    p.add_argument('--ridge_alpha', type=float, default=0., help='RF only: ridge on log-alpha')
    p.add_argument('--ridge_c', type=float, default=0., help='RF only: ridge on the intercept c')
    p.add_argument('--mu_c', type=float, default=0., help='RF only: prior mean for c')

    # output
    p.add_argument('--out_dir', default='cross_val_results', help='root directory for results')
    p.add_argument('--overwrite', action='store_true',
                   help='refit even if the result file already exists')
    p.add_argument('--dry_run', action='store_true',
                   help='tiny fitting budget for a fast end-to-end smoke test')
    args = p.parse_args()

    # resolve the family-dependent defaults left as None above
    family = model_family(args.model)
    if args.max_iters is None:
        args.max_iters = 2000 if family == 'RF' else 1000
    if args.ridge_lambda is None:
        args.ridge_lambda = 0.0 if family == 'RF' else RIDGE_LAMBDA
    if args.mu_alpha is None:
        args.mu_alpha = 0.0 if family == 'RF' else MU_ALPHA

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

    # the RF models carry none of the network structure (no depth / nonlinearity)
    if family == 'RF':
        inference_model, depth, nln_name, nln = None, None, None, None
    else:
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

    # THE CHOICE FLIP (RF only): the dataset codes 0 = right / 1 = left, but
    # reinforce_pipeline.RF_log_likelihood treats 1 as a rightward choice. Without this
    # the fitted right/left weights come out sign-mirrored. inputs is already
    # [bias, right contrast, left contrast], matching w = (bias, right, left), and
    # rewards are unaffected by the flip. The NN pipeline uses the native coding.
    if family == 'RF':
        outputs = jnp.asarray(flip_outputs(np.asarray(outputs)))

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

    spec = 'REINFORCE' if family == 'RF' else f'{depth}, {nln_name}'
    print(f'=== {args.dataset} mouse {mouse_id} | model {args.model} '
          f'({spec}) | fold {fold}/{args.K} ===', flush=True)
    print(f'T={T} trials | train={n_train} test={n_test} '
          f'(held-out fraction {n_test/T:.3f})', flush=True)

    t0 = time.time()
    result = fit_one_fold(inputs, outputs, rewards, train_mask, args.model, depth, nln,
                          inference_model, jr.PRNGKey(args.fit_seed), args)
    elapsed = time.time() - t0

    if result['test_nll'] is not None:
        print(f"\ntrain NLL={result['train_nll']:.3f} | TEST NLL={result['test_nll']:.3f} "
              f"| test NLL/trial={result['test_nll']/n_test:.5f} "
              f"(chance={np.log(2):.5f})", flush=True)
    print(f'converged={result["converged"]} '
          f'elapsed={elapsed:.1f}s', flush=True)

    record = dict(
        # identity
        dataset=args.dataset, mouse_id=mouse_id, model=args.model, family=family,
        inference_model=inference_model, depth=depth, nln_name=nln_name, saturate_inputs=args.saturate_inputs,
        flipped_outputs=(family == 'RF'),
        # fold metadata (enough to reproduce the split exactly)
        fold=fold, K=args.K, block_length=args.block_length, fold_seed=args.fold_seed,
        train_mask=np.asarray(train_mask), num_trials=T, n_train=n_train, n_test=n_test,
        # fit settings (only the active family's penalty terms are meaningful)
        num_seeds=args.num_seeds, max_iters=args.max_iters, tol=args.tol, fit_seed=args.fit_seed,
        ridge_lambda=args.ridge_lambda, mu_alpha=args.mu_alpha,
        alphabeta_ridge=args.alphabeta_ridge, mu_beta=args.mu_beta,
        ridge_alpha=args.ridge_alpha, ridge_c=args.ridge_c, mu_c=args.mu_c,
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
