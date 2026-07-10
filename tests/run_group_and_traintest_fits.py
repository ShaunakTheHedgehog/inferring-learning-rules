"""Integration / smoke tests for the two newer inference workflows in
`inference_pipeline.py`:

  (A) single-animal TRAIN/TEST-SPLIT fit  -- fit on ~90% of a mouse's trials
      (within-sequence block mask) and evaluate held-out ~10% test NLL.
  (B) GROUP (batched, multi-animal) fit   -- fit shared hyperparameters across
      several padded, variable-length animals at once.

These are kept deliberately small (few seeds / iters) so they run in ~1 min.
They print parameter-recovery diagnostics and assert basic sanity (finite NLLs,
positive alpha/beta, held-out test NLL close to the ground-truth model's).

Run:  .venv/bin/python tests/run_group_and_traintest_fits.py
      (or import and call run_train_test_split_fit() / run_group_fit())

NOTE: both workflows here avoid the known `compute_nll` unmasked bug -- (A) uses
masked=True and (B) uses batched=True, neither of which hits the buggy branch.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from utils import get_nonlinearity, make_train_mask
from inference_utils import convert_weights_to_dict, train_gd_network
from metric_GD_network import generate_randomized_multisession_graded_data
from inference_pipeline import (
    run_randomized_optimization_procedure,
    run_train_test_single_learning_forward_trajectory,
    batched_run_forward_learning_trajectory,
)

# --------------------------------------------------------------------------- #
# Shared ground truth
# --------------------------------------------------------------------------- #
NLN_NAME = "erf"
NLN = get_nonlinearity(NLN_NAME)
INFERENCE_MODEL = "gd"
DEPTH = "two-layer"
NUM_GRADINGS = 4
NSPG = 30            # samples per grading level
NZC = 10             # zero-contrast trials per session
TRUE_ALPHA = 0.005
TRUE_BETA = 8.0
TRUE_W0_ARR = jnp.array([0.5, 0.3, 0.1, 0.2, -0.1, 0.05, -0.3, 0.4, -0.2])
TRUE_HP = jnp.concatenate((jnp.log(jnp.array([TRUE_ALPHA, TRUE_BETA])), TRUE_W0_ARR))


def _simulate_animal(num_sessions, key):
    """Generate inputs and simulate one animal's choices/rewards from the GT net."""
    k_in, k_tr = jr.split(key, 2)
    inputs, targets, _ = generate_randomized_multisession_graded_data(
        num_sessions, NUM_GRADINGS, NSPG, NZC, k_in
    )
    _, _, rd = train_gd_network(
        NLN_NAME, convert_weights_to_dict(TRUE_W0_ARR), inputs, targets,
        TRUE_ALPHA, TRUE_BETA, k_tr, num_grading_levels=NUM_GRADINGS
    )
    outputs = jnp.array(rd["outputs"])
    rewards = jnp.array(rd["rewards"])
    return inputs, outputs, rewards


def _report_recovery(best_hp):
    num_alphas = 1  # 'gd' two-layer
    print(f"  recovered alpha = {float(jnp.exp(best_hp[num_alphas-1])):.5f}  (true {TRUE_ALPHA})")
    print(f"  recovered beta  = {float(jnp.exp(best_hp[num_alphas])):.4f}  (true {TRUE_BETA})")


# --------------------------------------------------------------------------- #
# (A) Single-animal train/test-split fit
# --------------------------------------------------------------------------- #
def run_train_test_split_fit(num_sessions=25, num_seeds=4, max_iters=400, seed=0):
    print("\n" + "=" * 70)
    print("(A) SINGLE-ANIMAL TRAIN/TEST-SPLIT FIT (fit on ~90%, eval on held-out ~10%)")
    print("=" * 70)
    k_data, k_fit = jr.split(jr.PRNGKey(seed), 2)
    inputs, outputs, rewards = _simulate_animal(num_sessions, k_data)
    T = int(outputs.shape[0])

    # within-sequence block mask (True = train). Seed numpy for reproducibility
    # since make_train_mask uses the global `random` module.
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    train_mask = jnp.array(make_train_mask(T, block_length=20, test_fraction=0.1))
    frac_test = float(jnp.mean(~train_mask))
    print(f"  T = {T} trials | held-out test fraction = {frac_test:.3f}")

    best_w0, min_train_obj, best_hp, optim = run_randomized_optimization_procedure(
        inputs, outputs, rewards, k_fit, DEPTH, NLN, INFERENCE_MODEL,
        batched=False, masked=True, train_mask=train_mask,
        num_seeds=num_seeds, max_iters=max_iters, tol=1e-4,
        alpha_init_scale=1e-1, beta_init_scale=1e1, w0_init_scale=1e-1,
    )

    # Evaluate train & test NLL for both the fitted and the ground-truth params
    inf_tr, inf_te, _, _ = run_train_test_single_learning_forward_trajectory(
        best_hp, inputs, outputs, rewards, train_mask, NLN, INFERENCE_MODEL, None)
    true_tr, true_te, _, _ = run_train_test_single_learning_forward_trajectory(
        TRUE_HP, inputs, outputs, rewards, train_mask, NLN, INFERENCE_MODEL, None)

    _report_recovery(best_hp)
    print(f"  train NLL: inferred {float(inf_tr):.2f} | true {float(true_tr):.2f}")
    print(f"  TEST  NLL: inferred {float(inf_te):.2f} | true {float(true_te):.2f}")

    # sanity assertions
    assert best_hp is not None, "no converged seed"
    assert np.isfinite(float(inf_te)) and np.isfinite(float(inf_tr))
    assert float(jnp.exp(best_hp[0])) > 0 and float(jnp.exp(best_hp[1])) > 0
    # fitted train NLL should be <= true params' train NLL (it was optimized on train)
    assert float(inf_tr) <= float(true_tr) + 1.0, "fit did not reach ground-truth train NLL"
    # held-out test NLL should be in the ballpark of the ground-truth model's
    assert float(inf_te) <= float(true_te) * 1.5 + 10.0, "held-out test NLL implausibly large"
    print("  [OK] train/test-split fit sanity checks passed")
    return best_hp, float(inf_tr), float(inf_te)


# --------------------------------------------------------------------------- #
# (B) Group (batched, multi-animal) fit
# --------------------------------------------------------------------------- #
def run_group_fit(all_num_sessions=(20, 30, 25), num_seeds=3, max_iters=400, seed=1):
    print("\n" + "=" * 70)
    print("(B) GROUP (BATCHED, MULTI-ANIMAL) FIT -- shared hyperparameters")
    print("=" * 70)
    key = jr.PRNGKey(seed)

    seqs = []
    for ns in all_num_sessions:
        key, k = jr.split(key)
        seqs.append(_simulate_animal(int(ns), k))
    lengths = jnp.array([s[1].shape[0] for s in seqs])
    T_max = int(lengths.max())
    print(f"  {len(seqs)} animals | lengths = {np.asarray(lengths)} | padded to {T_max}")

    def _pad(a, n):
        pad_width = [(0, n - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
        return jnp.pad(a, pad_width)

    all_inputs = jnp.stack([_pad(s[0], T_max) for s in seqs])
    all_outputs = jnp.stack([_pad(s[1], T_max) for s in seqs])
    all_rewards = jnp.stack([_pad(s[2], T_max) for s in seqs])

    _, _, k_fit = jr.split(key, 3)
    best_w0, min_obj, best_hp, optim = run_randomized_optimization_procedure(
        all_inputs, all_outputs, all_rewards, k_fit, DEPTH, NLN, INFERENCE_MODEL,
        batched=True, lengths=lengths,
        num_seeds=num_seeds, max_iters=max_iters, tol=1e-4,
        alpha_init_scale=1e-1, beta_init_scale=1e1, w0_init_scale=1e-1,
    )

    true_nlls, _, _ = batched_run_forward_learning_trajectory(
        TRUE_HP, all_inputs, all_outputs, all_rewards, lengths, NLN, INFERENCE_MODEL, None)
    inf_nlls, _, _ = batched_run_forward_learning_trajectory(
        best_hp, all_inputs, all_outputs, all_rewards, lengths, NLN, INFERENCE_MODEL, None)

    _report_recovery(best_hp)
    print(f"  per-animal NLL (inferred): {np.round(np.asarray(inf_nlls), 2)}")
    print(f"  per-animal NLL (true)    : {np.round(np.asarray(true_nlls), 2)}")

    assert best_hp is not None, "no converged seed"
    assert np.all(np.isfinite(np.asarray(inf_nlls)))
    assert float(jnp.exp(best_hp[0])) > 0 and float(jnp.exp(best_hp[1])) > 0
    # fitted mean NLL should be <= ground-truth mean NLL (fit optimizes the batch mean)
    assert float(jnp.mean(inf_nlls)) <= float(jnp.mean(true_nlls)) + 1.0
    print("  [OK] group-fit sanity checks passed")
    return best_hp, np.asarray(inf_nlls)


if __name__ == "__main__":
    run_train_test_split_fit()
    run_group_fit()
    print("\nAll integration workflows completed.")
