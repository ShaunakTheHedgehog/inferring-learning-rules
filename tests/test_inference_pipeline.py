"""Unit tests for the learning-rule inference pipeline.

Scope (per request): focus on `inference_pipeline.py` and `inference_utils.py` --
NLL correctness, hyperparameter packing/unpacking round-trips, train/test
masking, batched-vs-single equivalence, padding invariance, and the ridge prior.

Run with:   .venv/bin/python -m pytest tests/ -v
(conftest.py puts the repo root on sys.path.)

Notes
-----
* One test, `test_compute_nll_unmasked_default`, is marked xfail because it
  documents a *currently-live* bug: `compute_nll(..., batched=False)` with the
  default `masked=False, train_mask=None` raises TypeError, because `masked` is
  not a static JIT arg so `lax.cond` traces the train branch with
  `train_mask=None`. Remove the xfail once the pipeline is fixed (make `masked`
  static and use a Python `if`). This is the exact call made by
  `mouse_inference_data_analysis.get_mouse_aics` (line 101).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import pytest

from utils import get_nonlinearity
from inference_utils import (
    convert_to_dict, convert_to_array, convert_weights_to_dict,
    extract_weights_from_hyperparams, get_depth, get_num_alphas,
    generate_init_hyperparams, unpack_hyperparams,
    model_forward, log_likelihood, squared_error_loss_fn, train_gd_network,
)
from metric_GD_network import generate_randomized_multisession_graded_data
from inference_pipeline import (
    run_single_learning_forward_trajectory,
    run_train_test_single_learning_forward_trajectory,
    batched_run_forward_learning_trajectory,
    compute_nll,
)

ATOL = 1e-8


# --------------------------------------------------------------------------- #
# Fixtures: a small synthetic "animal" generated from a known ground-truth net
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_animal():
    """One synthetic two-layer erf-GD animal with known hyperparameters."""
    nln_name = "erf"
    nln = get_nonlinearity(nln_name)
    k_data, k_train = jr.split(jr.PRNGKey(0), 2)

    num_sessions, num_gradings, nspg, nzc = 5, 4, 15, 6
    inputs, targets, _ = generate_randomized_multisession_graded_data(
        num_sessions, num_gradings, nspg, nzc, k_data
    )

    alpha, beta = 0.01, 6.0
    w0_arr = jnp.array([0.5, 0.3, 0.1, 0.2, -0.1, 0.05, -0.3, 0.4, -0.2])
    w0 = convert_weights_to_dict(w0_arr)
    _, _, rd = train_gd_network(
        nln_name, w0, inputs, targets, alpha, beta, k_train, num_grading_levels=num_gradings
    )
    outputs = jnp.array(rd["outputs"])
    rewards = jnp.array(rd["rewards"])
    hp = jnp.concatenate((jnp.log(jnp.array([alpha, beta])), w0_arr))

    return dict(
        nln=nln, nln_name=nln_name, inference_model="gd", depth="two-layer",
        inputs=inputs, outputs=outputs, rewards=rewards, hp=hp,
        alpha=alpha, beta=beta, w0_arr=w0_arr, T=int(outputs.shape[0]),
    )


@pytest.fixture(scope="module")
def train_mask(synthetic_animal):
    """A reproducible ~90/10 boolean train mask (True = train)."""
    rng = np.random.default_rng(0)
    return jnp.array(rng.random(synthetic_animal["T"]) > 0.1)


# --------------------------------------------------------------------------- #
# Hyperparameter layout: pack / unpack / convert round-trips
# --------------------------------------------------------------------------- #
class TestHyperparamLayout:
    @pytest.mark.parametrize("depth,inference_model", [
        ("two-layer", "gd"),
        ("one-layer", "gd"),
        ("two-layer", "full_metric_gd"),
        ("two-layer", "pathway_metric_gd"),
    ])
    def test_dict_array_roundtrip(self, depth, inference_model):
        hp = generate_init_hyperparams(inference_model, depth, 1e-1, 1e0, 1e-1, jr.PRNGKey(3))
        d = convert_to_dict(hp, inference_model)
        hp2 = convert_to_array(d, inference_model)
        assert hp2.shape == hp.shape
        assert np.allclose(np.asarray(hp), np.asarray(hp2), atol=ATOL)

    @pytest.mark.parametrize("depth,inference_model,expected_alphas", [
        ("two-layer", "gd", 1),
        ("one-layer", "gd", 1),
        ("two-layer", "full_metric_gd", 3),
        ("two-layer", "pathway_metric_gd", 2),
        ("two-layer", "fixed_metric_gd", 2),
        ("one-layer", "full_metric_gd", 2),
    ])
    def test_get_num_alphas(self, depth, inference_model, expected_alphas):
        assert get_num_alphas(inference_model, depth) == expected_alphas

    def test_get_depth_from_length(self):
        hp_two = generate_init_hyperparams("gd", "two-layer", 1e-1, 1e0, 1e-1, jr.PRNGKey(1))
        hp_one = generate_init_hyperparams("gd", "one-layer", 1e-1, 1e0, 1e-1, jr.PRNGKey(1))
        assert get_depth(hp_two, "gd") == "two-layer"
        assert get_depth(hp_one, "gd") == "one-layer"

    def test_convert_weights_to_dict_two_layer(self):
        w = jnp.arange(9.0)
        d = convert_weights_to_dict(w)
        assert set(d.keys()) == {"W1_diag", "W2"}
        assert d["W1_diag"].shape == (3,)
        assert d["W2"].shape == (2, 3)
        assert np.allclose(np.asarray(d["W1_diag"]), [0, 1, 2])

    def test_convert_weights_to_dict_one_layer(self):
        w = jnp.arange(6.0)
        d = convert_weights_to_dict(w)
        assert set(d.keys()) == {"W"}
        assert d["W"].shape == (2, 3)

    def test_extract_weights_matches_unpack(self, synthetic_animal):
        hp = synthetic_animal["hp"]
        w_extract = extract_weights_from_hyperparams(hp, "gd", return_dict=True)
        _, _, w_unpack = unpack_hyperparams(hp, "gd")
        assert np.allclose(np.asarray(w_extract["W1_diag"]), np.asarray(w_unpack["W1_diag"]))
        assert np.allclose(np.asarray(w_extract["W2"]), np.asarray(w_unpack["W2"]))

    def test_unpack_hyperparams_values(self, synthetic_animal):
        hp = synthetic_animal["hp"]
        log_lrs, log_beta, w0 = unpack_hyperparams(hp, "gd")
        # gd shares a single alpha across both layers
        assert np.isclose(float(log_lrs["W1_diag"]), float(hp[0]))
        assert np.isclose(float(log_beta), float(hp[1]))
        assert np.allclose(np.asarray(w0["W1_diag"]), np.asarray(hp[2:5]))


# --------------------------------------------------------------------------- #
# Low-level model pieces
# --------------------------------------------------------------------------- #
class TestModelPieces:
    def test_model_forward_shapes(self):
        x = jnp.array([1.0, 0.5, 0.0])
        w2 = {"W1_diag": jnp.ones(3), "W2": jnp.ones((2, 3))}
        assert model_forward(w2, x, get_nonlinearity("erf")).shape == (2,)
        w1 = {"W": jnp.ones((2, 3))}
        assert model_forward(w1, x, get_nonlinearity("linear")).shape == (2,)

    def test_one_layer_forward_is_linear(self):
        x = jnp.array([1.0, 0.3, 0.7])
        W = jnp.array([[0.2, -0.1, 0.4], [0.5, 0.6, -0.3]])
        out = model_forward({"W": W}, x, get_nonlinearity("linear"))
        assert np.allclose(np.asarray(out), np.asarray(W @ x))

    def test_log_likelihood_is_nonpositive_and_monotone(self):
        # ll = log sigmoid(beta * (Q_chosen - Q_other)); increases as the gap grows
        log_beta = jnp.log(jnp.array(2.0))
        out_small = jnp.array([1.0, 0.9])   # small advantage for choice 0
        out_big = jnp.array([1.0, -1.0])    # large advantage for choice 0
        ll_small = float(log_likelihood(out_small, 0, log_beta))
        ll_big = float(log_likelihood(out_big, 0, log_beta))
        assert ll_small <= 0.0 and ll_big <= 0.0
        assert ll_big > ll_small

    def test_log_likelihood_symmetry(self):
        # p(choice 0 | gap g) == p(choice 1 | gap -g)
        log_beta = jnp.log(jnp.array(3.0))
        out = jnp.array([0.7, 0.2])
        ll0 = float(log_likelihood(out, 0, log_beta))
        ll1 = float(log_likelihood(out[::-1], 1, log_beta))
        assert np.isclose(ll0, ll1)

    def test_squared_error_loss_nonnegative(self, synthetic_animal):
        sa = synthetic_animal
        _, _, w0 = unpack_hyperparams(sa["hp"], "gd")
        loss = squared_error_loss_fn(w0, sa["inputs"][0], int(sa["outputs"][0]),
                                     float(sa["rewards"][0]), sa["nln"])
        assert float(loss) >= 0.0


# --------------------------------------------------------------------------- #
# Forward trajectory NLL
# --------------------------------------------------------------------------- #
class TestForwardTrajectory:
    def test_single_nll_finite_and_equals_sum_lls(self, synthetic_animal):
        sa = synthetic_animal
        nll, ws, lls = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        assert np.isfinite(float(nll))
        assert np.isclose(float(nll), float(-jnp.sum(lls)), atol=1e-6)

    def test_weight_trajectory_length(self, synthetic_animal):
        sa = synthetic_animal
        _, ws, _ = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        assert ws["W1_diag"].shape[0] == sa["T"]
        assert ws["W2"].shape[0] == sa["T"]

    def test_first_weight_update_uses_initial_weights(self, synthetic_animal):
        # The ll of trial 0 should be computed from w0 (pre-update), so the very
        # first log-likelihood must match a direct forward pass at w0.
        sa = synthetic_animal
        _, log_beta, w0 = unpack_hyperparams(sa["hp"], "gd")
        _, _, lls = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        out0 = model_forward(w0, sa["inputs"][0], sa["nln"])
        expected_ll0 = float(log_likelihood(out0, int(sa["outputs"][0]), log_beta))
        assert np.isclose(float(lls[0]), expected_ll0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Train/test masking
# --------------------------------------------------------------------------- #
class TestTrainTestMasking:
    def test_train_plus_test_equals_total(self, synthetic_animal, train_mask):
        sa = synthetic_animal
        total, _, _ = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        tr, te, _, _ = run_train_test_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], train_mask, sa["nln"], "gd", None)
        assert np.isclose(float(tr) + float(te), float(total), atol=1e-6)

    def test_all_train_mask_gives_total_and_zero_test(self, synthetic_animal):
        sa = synthetic_animal
        total, _, _ = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        full_mask = jnp.ones(sa["T"], dtype=bool)
        tr, te, _, _ = run_train_test_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], full_mask, sa["nln"], "gd", None)
        assert np.isclose(float(tr), float(total), atol=1e-6)
        assert np.isclose(float(te), 0.0, atol=1e-6)

    def test_mask_does_not_change_learning_trajectory(self, synthetic_animal, train_mask):
        # The mask only splits the likelihood; the simulated weight trajectory
        # (which learns from every trial) must be identical.
        sa = synthetic_animal
        _, _, lls_full = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        _, _, _, lls_masked = run_train_test_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], train_mask, sa["nln"], "gd", None)
        assert np.allclose(np.asarray(lls_full), np.asarray(lls_masked), atol=1e-6)


# --------------------------------------------------------------------------- #
# Batched / group path
# --------------------------------------------------------------------------- #
class TestBatched:
    def test_batched_single_seq_equals_non_batched(self, synthetic_animal):
        sa = synthetic_animal
        total, _, _ = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        nlls, _, _ = batched_run_forward_learning_trajectory(
            sa["hp"], sa["inputs"][None], sa["outputs"][None], sa["rewards"][None],
            jnp.array([sa["T"]]), sa["nln"], "gd", None)
        assert np.isclose(float(nlls[0]), float(total), atol=1e-6)

    def test_padding_invariance(self, synthetic_animal):
        # Padding then setting `lengths` to the true length must not change NLL.
        sa = synthetic_animal
        total, _, _ = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        pad = 37
        inp = jnp.pad(sa["inputs"], ((0, pad), (0, 0)))[None]
        out = jnp.pad(sa["outputs"], (0, pad))[None]
        rew = jnp.pad(sa["rewards"], (0, pad))[None]
        nlls, _, _ = batched_run_forward_learning_trajectory(
            sa["hp"], inp, out, rew, jnp.array([sa["T"]]), sa["nln"], "gd", None)
        assert np.isclose(float(nlls[0]), float(total), atol=1e-6)

    def test_variable_length_batch(self, synthetic_animal):
        # Two animals of different (true) length in one padded batch: each NLL
        # should match its own single-sequence NLL.
        sa = synthetic_animal
        T = sa["T"]
        half = T // 2
        # animal A: full; animal B: first `half` trials only
        totA, _, _ = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        totB, _, _ = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"][:half], sa["outputs"][:half], sa["rewards"][:half],
            sa["nln"], "gd", None)
        inp = jnp.stack([sa["inputs"], sa["inputs"]])
        out = jnp.stack([sa["outputs"], sa["outputs"]])
        rew = jnp.stack([sa["rewards"], sa["rewards"]])
        nlls, _, _ = batched_run_forward_learning_trajectory(
            sa["hp"], inp, out, rew, jnp.array([T, half]), sa["nln"], "gd", None)
        assert np.isclose(float(nlls[0]), float(totA), atol=1e-6)
        assert np.isclose(float(nlls[1]), float(totB), atol=1e-6)


# --------------------------------------------------------------------------- #
# compute_nll: prior, masked path, group path, and the known unmasked bug
# --------------------------------------------------------------------------- #
class TestComputeNLL:
    def test_masked_equals_train_nll(self, synthetic_animal, train_mask):
        sa = synthetic_animal
        tr, _, _, _ = run_train_test_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], train_mask, sa["nln"], "gd", None)
        nll = compute_nll(sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd",
                          False, None, True, train_mask, None, 9, 0., 0., 0., 0.)
        assert np.isclose(float(nll), float(tr), atol=1e-6)

    def test_batched_group_nll_is_mean(self, synthetic_animal):
        sa = synthetic_animal
        total, _, _ = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        inp = jnp.stack([sa["inputs"], sa["inputs"]])
        out = jnp.stack([sa["outputs"], sa["outputs"]])
        rew = jnp.stack([sa["rewards"], sa["rewards"]])
        nll = compute_nll(sa["hp"], inp, out, rew, sa["nln"], "gd", True,
                          jnp.array([sa["T"], sa["T"]]), False, None, None, 9, 0., 0., 0., 0.)
        # mean over two identical animals == single-animal NLL
        assert np.isclose(float(nll), float(total), atol=1e-6)

    def test_ridge_prior_adds_expected_amount(self, synthetic_animal, train_mask):
        # Compare masked NLL with and without ridge; the difference must equal
        # the closed-form prior term.
        sa = synthetic_animal
        hp = sa["hp"]
        N = len(hp)
        nw = 9
        ridge_lambda, ab_ridge, mu_a, mu_b = 0.5, 0.3, -6.9, 0.0
        base = compute_nll(hp, sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd",
                           False, None, True, train_mask, None, nw, 0., 0., 0., 0.)
        with_prior = compute_nll(hp, sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd",
                                 False, None, True, train_mask, None, nw,
                                 ridge_lambda, ab_ridge, mu_a, mu_b)
        expected_prior = (
            ridge_lambda * float(jnp.sum(hp[-nw:] ** 2))
            + ab_ridge * float(jnp.sum((hp[:N - nw - 1] - mu_a) ** 2))
            + ab_ridge * float((hp[N - nw - 1] - mu_b) ** 2)
        )
        assert np.isclose(float(with_prior) - float(base), expected_prior, atol=1e-6)

    def test_compute_nll_unmasked_default(self, synthetic_animal):
        # Regression guard for the previously-broken unmasked path (batched=False,
        # masked=False, train_mask=None) -- the exact call made by get_mouse_aics.
        # Must equal the plain single-animal NLL.
        sa = synthetic_animal
        total, _, _ = run_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
        nll = compute_nll(sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd",
                          False, None, False, None, None, 9, 0., 0., 0., 0.)
        assert np.isclose(float(nll), float(total), atol=1e-6)


# --------------------------------------------------------------------------- #
# Parameter recovery (slow-ish integration check; still small)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_parameter_recovery_reaches_ground_truth(synthetic_animal):
    """A short multi-seed fit on the training portion should reach a train NLL at
    least as low as the ground-truth parameters' train NLL.

    We seed Python's `random` and numpy because `make_train_mask` uses the global
    `random` module (so it is otherwise order-dependent across tests), and we pick
    the best seed straight from optim_dict rather than the convergence-gated return
    value, so the test does not hinge on any single seed formally converging.
    """
    import random as _random
    from inference_pipeline import run_randomized_optimization_procedure
    from utils import make_train_mask
    sa = synthetic_animal

    _random.seed(0)
    np.random.seed(0)
    tm = jnp.array(make_train_mask(sa["T"], block_length=20, test_fraction=0.1))

    _, _, _, optim = run_randomized_optimization_procedure(
        sa["inputs"], sa["outputs"], sa["rewards"], jr.PRNGKey(7), "two-layer", sa["nln"], "gd",
        batched=False, masked=True, train_mask=tm, num_seeds=4, max_iters=400, tol=1e-4,
    )
    nlls = np.asarray(optim["nlls"])
    assert np.any(np.isfinite(nlls)), "every optimization seed produced a non-finite objective"

    best_idx = int(np.nanargmin(nlls))
    best_hp = optim["hyperparams"][best_idx]
    # recovered alpha/beta must be positive and finite
    assert float(jnp.exp(best_hp[0])) > 0
    assert float(jnp.exp(best_hp[1])) > 0

    # the best fit should reach (roughly) the ground-truth train NLL
    true_train = float(compute_nll(
        sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd",
        False, None, True, tm, None, 9, 0., 0., 0., 0.))
    assert float(nlls[best_idx]) <= true_train + 2.0, (
        f"best fitted train NLL {float(nlls[best_idx]):.2f} did not reach "
        f"ground-truth train NLL {true_train:.2f}")
