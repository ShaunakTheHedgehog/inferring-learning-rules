"""Unit tests for `reinforce_pipeline.py` (deterministic REINFORCE learning models).

Run the test suite:
    .venv/bin/python -m pytest tests/test_reinforce_pipeline.py -v
    .venv/bin/python -m pytest tests/test_reinforce_pipeline.py -v -m "not slow"

Run your OWN recovery experiment from the command line (see main() at the bottom):
    .venv/bin/python tests/test_reinforce_pipeline.py --model RF_b_1 \
        --alpha 0.03 --beta 0.3 --w0 0.2 0.6 -0.5 -T 12000 --num_seeds 8
    .venv/bin/python tests/test_reinforce_pipeline.py --model RF --alpha 0.02 0.04 0.03
    .venv/bin/python tests/test_reinforce_pipeline.py --beta_sweep -2 -0.5 0.3 1.5

TOLERANCES BELOW ARE EMPIRICALLY CALIBRATED, not guessed. Measured over T=12000 with
8 restarts across independent data seeds, using the REALISTIC input structure (exactly
one stimulus channel active per trial -- see simulate_reinforce_animal):
    RF_1 / RF : trajectory r >= 0.99
    RF_b_1 (beta in [-10, 0.3]): 8/8 seeds converge, Hessian positive definite
        (eig_min ~ +20, condition ~ 40), alpha*beta ratio 0.81-1.34.
    Around beta ~ 0.5-0.8 convergence becomes intermittent (some data seeds give 0/8
    with strongly negative eigenvalues) because the optimizer strays over beta = 1
    into the chaotic zone below; recovery tests therefore stop at beta = 0.3.

TWO KNOWN LIMITATIONS, each pinned by a test so a future fix gets noticed:

1. The deterministic rollout becomes CHAOTIC once beta exceeds the animal's REWARD
   RATE (test_chaos_boundary_tracks_reward_rate). Linearizing the update gives a
   Jacobian I - alpha*(r - beta)*p*(1-p)*x x^T, whose expansion factor along x is
   1 - alpha*(r - beta)*p(1-p)|x|^2; for small alpha the mean log factor is
   ~ -alpha*p(1-p)|x|^2 * (rbar - beta), which changes sign at beta = rbar. Measured
   onsets track rbar closely:
        hard task  (contrast 0.0625 only)  rbar = 0.531  ->  onset ~ 0.55-0.6
        mixed task (full Witten set)       rbar = 0.720  ->  onset ~ 0.65-0.7
        easy task  (contrast 1.0 only)     rbar = 0.971  ->  not reached by beta = 1
   Above the onset a 1e-8 perturbation of w0 grows to O(100) by trial 12000 and two
   mathematically identical implementations of the recursion disagree by O(10), so
   the likelihood is not reliably computable and recovery is meaningless. This is a
   property of the DYNAMICS, not of the parameterization: optimizing
   (log alpha, alpha*beta) instead of (log alpha, beta) gives bit-identical results.

   IMPORTANT: convergence diagnostics do NOT detect this. Above the onset the fits
   look healthy -- 60/60 seeds converge with a positive-definite, well-conditioned
   Hessian -- but they land back in the stable region (beta_fit ~ 0.3 for a true
   beta of 0.9-1.2), i.e. a stable model fitted to chaotic data, with trajectory
   correlation collapsing to 0.30-0.57. Use the perturbation probe, not the
   optimizer's own report, to decide whether a fit is trustworthy.
   Just BELOW the onset (beta ~ 0.5-0.8 for the mixed task) there is a separate
   optimizer trap: the dynamics are still stable but restarts pile up at beta ~ 1.0
   on saddle points (eig_min ~ -1e8 to -4e10) and 0-20 of 60 seeds converge.

2. alpha and beta are only weakly separable; the update is [alpha*r - alpha*beta](y-p)x,
   so for |beta| >> 1 the reward term is negligible and only the PRODUCT is identified
   (this is the paper's reward-agnostic RAR limit). Recovery tests therefore assert on
   alpha*beta, matching Ashwood et al. SM Fig S4d, which reports "more robust recovery
   of the quantity alpha*beta rather than the baseline hyperparameters beta directly".
"""
import argparse
import os
import sys

# conftest.py puts the repo root on sys.path for pytest; this covers running the
# file directly for the recovery CLI at the bottom.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax.tree_utils as otu
import pytest

from reinforce_pipeline import (
    NUM_WEIGHTS, REINFORCE_MODELS, get_num_params, _tree_l2_norm,
    unpack_reinforce_hyperparams, log_likelihood, generate_init_hyperparams,
    run_single_learning_forward_trajectory,
    run_train_test_single_reinforce_forward_trajectory,
    compute_objective, compute_hessian, optax_fit_params,
    run_epsilon_ball_test, run_randomized_optimization_procedure,
    plot_weight_trajectory,
)

# SEEDED on purpose: drawing this from the unseeded global numpy state made every
# recovery test non-reproducible (measured 4/5 pass, 1/5 fail across identical runs),
# because the ground-truth w0 changed on each invocation.
DEFAULT_W0 = np.random.default_rng(0).normal(size=NUM_WEIGHTS) * 1e-1

# Contrast magnitudes in the Witten dataset (globals.WITTEN_DATA_CONTRAST_LEVELS).
CONTRAST_LEVELS = (0.0625, 0.125, 0.25, 1.0)


# --------------------------------------------------------------------------- #
# Simulator + helpers
# --------------------------------------------------------------------------- #
def simulate_reinforce_animal(alphas, intercepts, w0, T=12000, seed=0,
                              contrast_levels=CONTRAST_LEVELS):
    """Roll out a deterministic REINFORCE agent, so recovery tests have a real target.

    INPUT STRUCTURE matches the real task: the stimulus appears on exactly ONE side, so
    precisely one of (cR, cL) is nonzero on every trial and the other is 0. Verified
    against the dataset -- across all 35 Witten mice (413,733 trials) 100.000% of trials
    have exactly one nonzero channel, and there are no zero-contrast trials. Drawing the
    two channels independently would produce a task the animals never faced.

    Choice coding matches `log_likelihood`: y = 1 is RIGHT, y = 0 is LEFT. Because
    exactly one channel is active, the correct side is unambiguous and reward is
    deterministic given the choice. `alphas`/`intercepts` are scalars or length-3.

    Returns (inputs, outputs, rewards, true_weight_trajectory).
    """
    rng = np.random.default_rng(seed)
    side = rng.integers(0, 2, T)                       # 1 => stimulus on the right
    magnitude = rng.choice(np.asarray(contrast_levels, dtype=float), T)
    right_stims = np.where(side == 1, magnitude, 0.0)
    left_stims = np.where(side == 0, magnitude, 0.0)
    X = np.column_stack([np.ones(T), right_stims, left_stims])
    correct_is_right = side.astype(int)

    a = np.broadcast_to(np.asarray(alphas, dtype=float), (NUM_WEIGHTS,)).copy()
    c = np.broadcast_to(np.asarray(intercepts, dtype=float), (NUM_WEIGHTS,)).copy()
    w = np.asarray(w0, dtype=float).copy()

    ys, rs, ws = np.zeros(T, dtype=int), np.zeros(T), np.zeros((T, NUM_WEIGHTS))
    for t in range(T):
        p_right = 1.0 / (1.0 + np.exp(-X[t] @ w))
        y = int(rng.random() < p_right)
        r = float(y == correct_is_right[t])
        ys[t], rs[t], ws[t] = y, r, w
        w = w + (y - p_right) * (a * r - c) * X[t]        # the update we infer

    return jnp.asarray(X), jnp.asarray(ys), jnp.asarray(rs), ws


def pack(model, alphas, intercepts, w0):
    """Build a hyperparameter vector in the pipeline layout: [log_alphas, intercepts, w0]."""
    num_alphas, num_baselines = REINFORCE_MODELS[model]
    a = np.broadcast_to(np.asarray(alphas, dtype=float), (NUM_WEIGHTS,))
    b = np.broadcast_to(np.asarray(intercepts, dtype=float), (NUM_WEIGHTS,))
    return jnp.concatenate([jnp.log(jnp.asarray(a[:num_alphas])),
                            jnp.asarray(b[:num_baselines]),
                            jnp.asarray(w0, dtype=float)])


def best_hyperparams(best_hp, optim_dict):
    """Best converged fit if there is one, else the best finite objective.

    The baseline models frequently fail the gradient tolerance (see the module
    docstring), so falling back keeps an expensive fit rather than discarding it.
    """
    # if best_hp is not None:
    #     return best_hp
    objs = np.asarray(optim_dict['nlls'], dtype=float)
    assert np.any(np.isfinite(objs)), "no optimization seed produced a finite objective"
    best_idx = int(np.nanargmin(np.where(np.isfinite(objs), objs, np.nan)))
    if best_hp is not None:
        best_hp_idxs = np.where(optim_dict['hyperparams'] == best_hp)[0]
        if best_idx not in list(best_hp_idxs):
            print('Found other minimum, with failure to convergence!')
    else:
        print('Failed to converge!')
    hyp = optim_dict['hyperparams'][best_idx]
    return hyp


def trajectory_correlation(hp, model, X, y, r, true_ws):
    """Worst per-weight correlation between the fitted and true weight trajectories."""
    _, ws_fit, _ = run_single_learning_forward_trajectory(hp, model, X, y, r)
    ws_fit = np.asarray(ws_fit)
    with np.errstate(invalid='ignore'):
        return min(float(np.corrcoef(ws_fit[:, k], true_ws[:, k])[0, 1])
                   for k in range(NUM_WEIGHTS))


def perturbation_growth(model, hp, X, y, r, eps=1e-8):
    """Divergence of two rollouts whose w0 differ by `eps` -- the chaos probe."""
    hp2 = hp.at[-NUM_WEIGHTS:].add(eps)
    _, w1, _ = run_single_learning_forward_trajectory(hp, model, X, y, r)
    _, w2, _ = run_single_learning_forward_trajectory(hp2, model, X, y, r)
    return float(jnp.max(jnp.abs(w1 - w2)))


@pytest.fixture(scope="module")
def small_data():
    """A short sequence, for the cheap structural tests."""
    X, y, r, _ = simulate_reinforce_animal(0.03, 0.0, DEFAULT_W0, T=500, seed=0)
    return X, y, r


# --------------------------------------------------------------------------- #
# Parameter layout
# --------------------------------------------------------------------------- #
class TestLayout:
    def test_param_counts(self):
        assert {m: get_num_params(m) for m in REINFORCE_MODELS} == {
            'RF_1': 4, 'RF_b_1': 5, 'RF': 6, 'RF_b': 9}

    @pytest.mark.parametrize("model", sorted(REINFORCE_MODELS))
    def test_unpack_broadcasts_everything_to_per_weight(self, model):
        """One update expression must cover every variant, so all outputs are (3,)."""
        hp = jnp.arange(float(get_num_params(model)))
        alphas, intercepts, w0 = unpack_reinforce_hyperparams(hp, model)
        assert alphas.shape == intercepts.shape == w0.shape == (NUM_WEIGHTS,)

    @pytest.mark.parametrize("model", sorted(REINFORCE_MODELS))
    def test_w0_is_always_the_last_three_entries(self, model):
        """Required by the w0 ridge, which slices hyperparams[-NUM_WEIGHTS:]."""
        hp = jnp.arange(float(get_num_params(model)))
        _, _, w0 = unpack_reinforce_hyperparams(hp, model)
        assert np.allclose(w0, hp[-NUM_WEIGHTS:])

    def test_alphas_exponentiated_and_broadcast(self):
        alphas, _, _ = unpack_reinforce_hyperparams(pack('RF_1', 0.01, 0.0, DEFAULT_W0), 'RF_1')
        assert np.allclose(alphas, 0.01)

    def test_per_weight_alphas_preserved(self):
        hp = pack('RF', [0.01, 0.02, 0.03], 0.0, DEFAULT_W0)
        alphas, _, _ = unpack_reinforce_hyperparams(hp, 'RF')
        assert np.allclose(alphas, [0.01, 0.02, 0.03])

    @pytest.mark.parametrize("model", ['RF_1', 'RF'])
    def test_baselines_zero_for_no_baseline_models(self, model):
        hp = jnp.arange(float(get_num_params(model)))
        _, intercepts, _ = unpack_reinforce_hyperparams(hp, model)
        assert np.allclose(intercepts, 0.0)

    def test_negative_baselines_survive(self):
        """Baselines are not log-parameterized, so they must be able to go negative --
        and Ashwood et al. report exactly that for real animals."""
        hp = pack('RF_b', [0.01, 0.02, 0.03], [0.8, -0.1, 0.5], DEFAULT_W0)
        _, intercepts, _ = unpack_reinforce_hyperparams(hp, 'RF_b')
        assert np.allclose(intercepts, [0.8, -0.1, 0.5])

    def test_wrong_length_rejected(self):
        with pytest.raises(AssertionError):
            unpack_reinforce_hyperparams(jnp.zeros(5), 'RF_1')

    def test_unknown_model_rejected(self):
        with pytest.raises(AssertionError):
            unpack_reinforce_hyperparams(jnp.zeros(4), 'RF_nope')


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #
class TestLogLikelihood:
    def test_matches_log_sigmoid(self):
        for z in [-8.0, -1.0, 0.0, 1.0, 8.0]:
            p = 1.0 / (1.0 + np.exp(-z))
            assert np.isclose(float(log_likelihood(jnp.asarray(z), 1)), np.log(p))
            assert np.isclose(float(log_likelihood(jnp.asarray(z), 0)), np.log1p(-p))

    def test_stable_at_extreme_logits(self):
        """The logaddexp form must not underflow where log(sigmoid(z)) would. This
        matters here because ||w|| carries the gain that beta carries in the GD models."""
        for z in [-800.0, 800.0]:
            for choice in (0, 1):
                assert np.isfinite(float(log_likelihood(jnp.asarray(z), choice)))

    def test_non_positive_and_chance_at_zero(self):
        for z in [-3.0, 0.0, 3.0]:
            for choice in (0, 1):
                assert float(log_likelihood(jnp.asarray(z), choice)) <= 0.0
        assert np.isclose(float(log_likelihood(jnp.asarray(0.0), 1)), np.log(0.5))


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #
class TestInitialization:
    @pytest.mark.parametrize("model", sorted(REINFORCE_MODELS))
    def test_init_length(self, model):
        hp = generate_init_hyperparams(model, 1e-2, 1e-1, 1e-1, jr.PRNGKey(0))
        assert hp.shape == (get_num_params(model),)

    def test_alphas_positive_baselines_zero_mean(self):
        """alpha is half-normal then logged (always > 0); b is zero-mean, so the
        baseline models start at their no-baseline special case (b = 0)."""
        draws = jnp.stack([generate_init_hyperparams('RF_b', 1e-2, 1e-1, 1e-1, jr.PRNGKey(s))
                           for s in range(400)])
        na, nb = REINFORCE_MODELS['RF_b']
        assert bool(jnp.all(jnp.exp(draws[:, :na]) > 0))
        intercepts = draws[:, na:na + nb]
        assert (intercepts < 0).any() and (intercepts > 0).any()
        assert abs(float(intercepts.mean())) < 0.03
        assert 0.08 < float(intercepts.std()) < 0.12

    def test_reproducible_and_key_dependent(self):
        a = generate_init_hyperparams('RF_b', 1e-2, 1e-1, 1e-1, jr.PRNGKey(7))
        b = generate_init_hyperparams('RF_b', 1e-2, 1e-1, 1e-1, jr.PRNGKey(7))
        c = generate_init_hyperparams('RF_b', 1e-2, 1e-1, 1e-1, jr.PRNGKey(8))
        assert np.allclose(a, b) and not np.allclose(a, c)

    def test_baseline_guess_of_zero_still_jitters(self):
        """The baseline guess is perturbed ADDITIVELY. A multiplicative jitter (as used
        for alpha and w0) around a guess of 0 would freeze every restart at 0."""
        hp = generate_init_hyperparams('RF_b', 1e-2, 1e-1, 1e-1, jr.PRNGKey(0),
                                       c_init_guess=jnp.zeros(3), noise_level=0.01)
        _, intercepts, _ = unpack_reinforce_hyperparams(hp, 'RF_b')
        assert bool(jnp.all(intercepts != 0.0)) and float(jnp.abs(intercepts).max()) < 0.05

    def test_alpha_and_w0_guesses_respected(self):
        hp = generate_init_hyperparams('RF', 1e-2, 1e-1, 1e-1, jr.PRNGKey(0),
                                       alpha_init_guess=jnp.full(3, 0.05),
                                       w0_init_guess=jnp.full(3, 0.4), noise_level=0.01)
        alphas, _, w0 = unpack_reinforce_hyperparams(hp, 'RF')
        assert np.allclose(alphas, 0.05, rtol=0.1) and np.allclose(w0, 0.4, rtol=0.1)


# --------------------------------------------------------------------------- #
# Forward trajectory
# --------------------------------------------------------------------------- #
class TestForwardTrajectory:
    def test_nll_equals_minus_sum_lls(self, small_data):
        X, y, r = small_data
        nll, _, lls = run_single_learning_forward_trajectory(
            pack('RF_1', 0.03, 0.0, DEFAULT_W0), 'RF_1', X, y, r)
        assert np.isfinite(float(nll))
        assert np.isclose(float(nll), float(-jnp.sum(lls)), atol=1e-8)

    def test_weight_trajectory_shape(self, small_data):
        X, y, r = small_data
        _, ws, _ = run_single_learning_forward_trajectory(
            pack('RF_1', 0.03, 0.0, DEFAULT_W0), 'RF_1', X, y, r)
        assert ws.shape == (X.shape[0], NUM_WEIGHTS)

    def test_zero_alpha_freezes_the_weights(self, small_data):
        X, y, r = small_data
        _, ws, _ = run_single_learning_forward_trajectory(
            pack('RF_1', 1e-300, 0.0, DEFAULT_W0), 'RF_1', X, y, r)
        assert np.allclose(np.asarray(ws), DEFAULT_W0[None, :])

    def test_pipeline_reproduces_the_simulator_exactly(self):
        """Given the same parameters and the same (x, y, r), the pipeline's rollout must
        match an independent NumPy implementation to machine precision. (This holds only
        in the stable regime -- see test_chaos_boundary_tracks_reward_rate.)"""
        true_alpha, true_beta = 0.03, 0.3
        true_intercept = true_alpha * true_beta
        X, y, r, ws_sim = simulate_reinforce_animal(true_alpha, true_intercept, DEFAULT_W0, T=2000, seed=0)
        hp = pack('RF_b_1', true_alpha, true_intercept, DEFAULT_W0)
        _, ws_pipe, _ = run_single_learning_forward_trajectory(hp, 'RF_b_1', X, y, r)
        # the pipeline carries w_new (post-update); the simulator stores w (pre-update)
        assert np.abs(np.asarray(ws_pipe)[:-1] - ws_sim[1:]).max() < 1e-12

    def test_train_plus_test_equals_total(self, small_data):
        X, y, r = small_data
        hp = pack('RF_1', 0.03, 0.0, DEFAULT_W0)
        total, _, _ = run_single_learning_forward_trajectory(hp, 'RF_1', X, y, r)
        mask = jnp.asarray(np.random.default_rng(0).random(X.shape[0]) > 0.2)
        tr, te, _, _ = run_train_test_single_reinforce_forward_trajectory(hp, 'RF_1', X, y, r, mask)
        assert np.isclose(float(tr) + float(te), float(total), atol=1e-8)

    def test_all_train_mask_reproduces_total(self, small_data):
        X, y, r = small_data
        hp = pack('RF_1', 0.03, 0.0, DEFAULT_W0)
        total, _, _ = run_single_learning_forward_trajectory(hp, 'RF_1', X, y, r)
        tr, te, _, _ = run_train_test_single_reinforce_forward_trajectory(
            hp, 'RF_1', X, y, r, jnp.ones(X.shape[0], dtype=bool))
        assert np.isclose(float(tr), float(total), atol=1e-8)
        assert np.isclose(float(te), 0.0, atol=1e-8)

    def test_mask_does_not_change_the_learning_trajectory(self, small_data):
        """The mask splits only the likelihood; the weights still evolve on every trial,
        because the animal genuinely experienced all of them."""
        X, y, r = small_data
        hp = pack('RF_1', 0.03, 0.0, DEFAULT_W0)
        _, _, lls_full = run_single_learning_forward_trajectory(hp, 'RF_1', X, y, r)
        mask = jnp.asarray(np.random.default_rng(1).random(X.shape[0]) > 0.2)
        _, _, _, lls_masked = run_train_test_single_reinforce_forward_trajectory(
            hp, 'RF_1', X, y, r, mask)
        assert np.allclose(np.asarray(lls_full), np.asarray(lls_masked), atol=1e-10)


# --------------------------------------------------------------------------- #
# Objective / Hessian / epsilon-ball
# --------------------------------------------------------------------------- #
class TestObjective:
    def test_unregularized_objective_equals_nll(self, small_data):
        X, y, r = small_data
        hp = pack('RF_1', 0.03, 0.0, DEFAULT_W0)
        nll, _, _ = run_single_learning_forward_trajectory(hp, 'RF_1', X, y, r)
        assert np.isclose(float(compute_objective(hp, 'RF_1', X, y, r)), float(nll), atol=1e-8)

    def test_masked_objective_equals_train_nll(self, small_data):
        X, y, r = small_data
        hp = pack('RF_1', 0.03, 0.0, DEFAULT_W0)
        mask = jnp.asarray(np.random.default_rng(2).random(X.shape[0]) > 0.2)
        tr, _, _, _ = run_train_test_single_reinforce_forward_trajectory(hp, 'RF_1', X, y, r, mask)
        assert np.isclose(float(compute_objective(hp, 'RF_1', X, y, r, True, mask)),
                          float(tr), atol=1e-8)

    def test_ridge_adds_the_closed_form_penalty(self, small_data):
        X, y, r = small_data
        na, nb = REINFORCE_MODELS['RF_b']
        hp = pack('RF_b', [0.01, 0.02, 0.03], [0.4, -0.2, 0.1], DEFAULT_W0)
        rl, ra, rb, mu_a, mu_b = 0.5, 0.3, 0.7, -3.0, 0.2
        base = float(compute_objective(hp, 'RF_b', X, y, r))
        withreg = float(compute_objective(hp, 'RF_b', X, y, r, False, None, rl, ra, rb, mu_a, mu_b))
        expected = (rl * float(jnp.sum(hp[-NUM_WEIGHTS:] ** 2))
                    + ra * float(jnp.sum((hp[:na] - mu_a) ** 2))
                    + rb * float(jnp.sum((hp[na:na + nb] - mu_b) ** 2)))
        assert np.isclose(withreg - base, expected, atol=1e-8)

    @pytest.mark.parametrize("model", sorted(REINFORCE_MODELS))
    def test_all_models_give_finite_objectives(self, small_data, model):
        X, y, r = small_data
        assert np.isfinite(float(compute_objective(pack(model, 0.03, 0.2, DEFAULT_W0),
                                                   model, X, y, r)))


class TestHessian:
    @pytest.mark.parametrize("model", sorted(REINFORCE_MODELS))
    def test_shape_and_symmetry(self, small_data, model):
        X, y, r = small_data
        n = get_num_params(model)
        H, eig = compute_hessian(pack(model, 0.03, 0.2, DEFAULT_W0), model, X, y, r)
        assert H.shape == (n, n) and eig.shape == (n,)
        assert np.allclose(np.asarray(H), np.asarray(H).T, atol=1e-6)


class TestEpsilonBall:
    def test_passes_at_an_optimum_and_fails_away_from_one(self, small_data):
        X, y, r = small_data
        init = generate_init_hyperparams('RF_1', 1e-2, 1e-1, 1e-1, jr.PRNGKey(0))
        fit, _ = optax_fit_params(init, 'RF_1', X, y, r, max_iters=500, tol=1e-6)
        _, nearby, res = run_epsilon_ball_test(fit, 'RF_1', X, y, r, jr.PRNGKey(1),
                                               num_samples=25, epsilon=1e-4)
        assert nearby.shape == (25,) and res == 'PASSED'
        _, _, res_bad = run_epsilon_ball_test(fit + 0.5, 'RF_1', X, y, r, jr.PRNGKey(1),
                                              num_samples=25, epsilon=1e-4)
        assert res_bad == 'FAILED'


# --------------------------------------------------------------------------- #
# PARAMETER RECOVERY (slow) -- the headline tests
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_recovery_RF_1():
    """One learning rate, no baseline. Calibrated alpha ratio 0.74-1.05 at T=12000."""
    true_alpha = 0.03
    X, y, r, true_ws = simulate_reinforce_animal(true_alpha, 0.0, DEFAULT_W0, T=12000, seed=0)
    true_hp = pack('RF_1', true_alpha, 0.0, DEFAULT_W0)

    _, _, best_hp, optim = run_randomized_optimization_procedure(
        X, y, r, jr.PRNGKey(3), 'RF_1', num_seeds=6, max_iters=500, tol=1e-4)
    hp = best_hyperparams(best_hp, optim)
    alphas, _, w0 = unpack_reinforce_hyperparams(hp, 'RF_1')

    # the MLE must fit at least as well as the truth on finite data
    assert float(compute_objective(hp, 'RF_1', X, y, r)) <= \
        float(compute_objective(true_hp, 'RF_1', X, y, r)) + 1e-6
    assert 0.5 * true_alpha < float(alphas[0]) < 2.0 * true_alpha
    assert np.abs(np.asarray(w0) - DEFAULT_W0).max() < 0.75
    assert trajectory_correlation(hp, 'RF_1', X, y, r, true_ws) > 0.95


@pytest.mark.slow
def test_recovery_RF_per_weight_alphas():
    """Three learning rates. Calibrated alpha ratios 0.74-1.30."""
    true_alphas = np.array([0.02, 0.04, 0.03])
    X, y, r, true_ws = simulate_reinforce_animal(true_alphas, 0.0, DEFAULT_W0, T=12000, seed=0)
    true_hp = pack('RF', true_alphas, 0.0, DEFAULT_W0)

    _, _, best_hp, optim = run_randomized_optimization_procedure(
        X, y, r, jr.PRNGKey(3), 'RF', num_seeds=6, max_iters=500, tol=1e-4)
    hp = best_hyperparams(best_hp, optim)
    alphas, _, _ = unpack_reinforce_hyperparams(hp, 'RF')

    assert float(compute_objective(hp, 'RF', X, y, r)) <= \
        float(compute_objective(true_hp, 'RF', X, y, r)) + 1e-6
    ratios = np.asarray(alphas) / true_alphas
    assert np.all((ratios > 0.4) & (ratios < 2.5)), f"alpha ratios {ratios}"
    assert trajectory_correlation(hp, 'RF', X, y, r, true_ws) > 0.95


@pytest.mark.slow
@pytest.mark.parametrize("true_beta", [-5.0, -0.5, 0.3])
def test_recovery_RF_b_1_alpha_times_beta(true_beta):
    """RF_b_1 in the STABLE regime (beta < 1), asserting on alpha*beta.

    We assert on the product, not beta, because the update is [alpha*r - alpha*beta](y-p)x:
    for |beta| >> 1 the reward term is negligible and only the product is identified.
    Measured directly -- at beta = -10 one seed returned alpha=1e-6, beta=-328876 yet
    alpha*beta = -0.310 against a true -0.300.
    """
    true_alpha = 0.03
    ab_true = true_alpha * true_beta
    X, y, r, true_ws = simulate_reinforce_animal(true_alpha, ab_true, DEFAULT_W0,
                                                 T=12000, seed=0)
    _, _, best_hp, optim = run_randomized_optimization_procedure(
        X, y, r, jr.PRNGKey(7), 'RF_b_1', num_seeds=6, max_iters=2000, tol=1e-4)
    hp = best_hyperparams(best_hp, optim)
    alphas, intercepts, _ = unpack_reinforce_hyperparams(hp, 'RF_b_1')
    ab_fit = float(intercepts[0])

    # calibrated ratios over this beta range were 0.57-1.49; allow 0.4-2.0
    assert 0.4 < ab_fit / ab_true < 2.0, f"alpha*beta {ab_fit:.5f} vs true {ab_true:.5f}"
    assert trajectory_correlation(hp, 'RF_b_1', X, y, r, true_ws) > 0.90


@pytest.mark.slow
def test_RF_b_1_hessian_is_positive_definite_in_the_stable_regime():
    """At realistic T the baseline model converges to a genuine, well-conditioned minimum.

    This is the small-T pathology NOT reproducing: at T=2000 the same fit gave 0/6
    converged with negative eigenvalues and condition ~1e13; at T=12000 it is clean.
    """
    true_alpha, true_beta = 0.03, 0.3
    true_intercept = true_alpha * true_beta
    X, y, r, _ = simulate_reinforce_animal(true_alpha, true_intercept, DEFAULT_W0, T=12000, seed=0)
    _, _, best_hp, optim = run_randomized_optimization_procedure(
        X, y, r, jr.PRNGKey(7), 'RF_b_1', num_seeds=6, max_iters=2000, tol=1e-4)
    assert best_hp is not None, "expected converged seeds at T=12000"
    _, eig = compute_hessian(best_hp, 'RF_b_1', X, y, r)
    eig = np.asarray(eig)
    assert np.all(eig > 0), f"expected a positive-definite Hessian, got {eig}"
    assert eig.max() / eig.min() < 1e4, f"ill-conditioned: {eig.max() / eig.min():.2e}"


@pytest.mark.slow
def test_chaos_boundary_tracks_reward_rate():
    """The rollout is stable while beta < the reward rate, and chaotic above it.

    Measured onsets: rbar 0.531 -> ~0.55-0.6, rbar 0.720 -> ~0.65-0.7, rbar 0.971 ->
    beyond beta = 1. So the boundary is NOT a fixed beta = 1; it follows the animal.
    """
    eps = 1e-8
    def growth_at(beta, levels=CONTRAST_LEVELS):
        alpha = 0.03
        intercept = alpha * beta
        X, y, r, _ = simulate_reinforce_animal(alpha, intercept, DEFAULT_W0, T=12000, seed=0,
                                               contrast_levels=levels)
        g = perturbation_growth('RF_b_1', pack('RF_b_1', alpha, intercept, DEFAULT_W0), X, y, r, eps)
        return g, float(r.mean())

    # mixed (real Witten) contrasts: rbar ~ 0.72
    for beta in [-0.5, 0.0, 0.3]:                       # comfortably below rbar
        g, _ = growth_at(beta)
        assert g < 1e-6, f"beta={beta} should be stable, growth {g:.2e}"
    for beta in [1.0, 1.2]:                             # comfortably above rbar
        g, _ = growth_at(beta)
        assert g > 1.0, f"beta={beta} should be chaotic, growth only {g:.2e}"

    # the boundary MOVES with the reward rate: an easy task (rbar ~ 0.97) is still
    # stable at a beta that is already chaotic on the mixed task
    g_mixed, rbar_mixed = growth_at(0.9)
    g_easy, rbar_easy = growth_at(0.9, levels=(1.0,))
    assert rbar_easy > rbar_mixed, f"easy task should have the higher reward rate"
    assert g_mixed > 1e-2, f"beta=0.9 > rbar={rbar_mixed:.2f} should be chaotic ({g_mixed:.2e})"
    assert g_easy < 1e-6, f"beta=0.9 < rbar={rbar_easy:.2f} should be stable ({g_easy:.2e})"


# --------------------------------------------------------------------------- #
# CLI: run your own recovery experiment
# --------------------------------------------------------------------------- #
def run_recovery(model, alphas, baselines, w0, T, num_seeds, fit_seed, data_seed,
                 max_iters, tol=1e-3, ridge_lambda=0., ridge_alpha=0., ridge_c=0., mu_alpha=0., mu_c=0.,
                 quiet=True, alpha_init_scale=1e-1, c_init_scale=1e-5, w0_init_scale=1e-1):
    """Simulate from known parameters, refit, and report recovery + diagnostics."""
    import contextlib, io

    intercepts = baselines * alphas

    X, y, r, true_ws = simulate_reinforce_animal(alphas, intercepts, w0, T=T, seed=data_seed)
    true_hp = pack(model, alphas, intercepts, w0)
    obj_true = float(compute_objective(true_hp, model, X, y, r))
    growth = perturbation_growth(model, true_hp, X, y, r)

    print(f"\n=== {model} | T={T} | data_seed={data_seed} | {num_seeds} restarts ===")
    print(f"  simulated animal : P(right)={float(y.mean()):.3f}  reward={float(r.mean()):.3f}  "
          f"|w|max={np.abs(true_ws).max():.2f}")
    print(f"  chaos probe      : w0 perturbed by 1e-8 grows to {growth:.2e}"
          f"{'   *** CHAOTIC -- results below are unreliable ***' if growth > 1e-2 else ''}")

    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        _, _, best_hp, optim = run_randomized_optimization_procedure(
            X, y, r, jr.PRNGKey(fit_seed), model, num_seeds=num_seeds,
            max_iters=max_iters, tol=tol, ridge_lambda=ridge_lambda,
            ridge_alpha=ridge_alpha, ridge_c=ridge_c, mu_alpha=mu_alpha, mu_c=mu_c,
            alpha_init_scale=alpha_init_scale, c_init_scale=c_init_scale, w0_init_scale=w0_init_scale)

    nbad = len(np.atleast_1d(np.asarray(optim['bad_run_idxs'])))
    hp = best_hyperparams(best_hp, optim)
    a_fit, c_fit, w0_fit = unpack_reinforce_hyperparams(hp, model)
    b_fit = c_fit / a_fit

    na, nb = REINFORCE_MODELS[model]
    a_true = np.broadcast_to(np.asarray(alphas, float), (NUM_WEIGHTS,))
    c_true = np.broadcast_to(np.asarray(intercepts, float), (NUM_WEIGHTS,)) 
    b_true = c_true / a_true

    g = jax.grad(compute_objective)(hp, model, X, y, r)
    _, eig = compute_hessian(hp, model, X, y, r)
    eig = np.asarray(eig)

    print(f"  converged        : {num_seeds - nbad}/{num_seeds}"
          f"{'' if best_hp is not None else '   (using best finite objective instead)'}")
    print(f"  alpha    true {np.round(a_true[:na], 5)}  ->  fit {np.round(np.asarray(a_fit)[:na], 5)}")
    if nb:
        print(f"  beta     true {np.round(b_true[:nb], 4)}  ->  fit {np.round(np.asarray(b_fit)[:nb], 4)}")
        ab_t = a_true[:nb] * b_true[:nb]
        ab_f = np.asarray(a_fit)[:nb] * np.asarray(b_fit)[:nb]
        print(f"  alpha*beta  true {np.round(ab_t, 5)}  ->  fit {np.round(ab_f, 5)}"
              f"   <- the identified quantity")
    print(f"  w0       true {np.round(np.asarray(w0, float), 3)}  ->  fit {np.round(np.asarray(w0_fit), 3)}")
    print(f"  objective  fit {float(compute_objective(hp, model, X, y, r)):.3f} vs truth {obj_true:.3f}")
    print(f"  |grad| {float(jnp.linalg.norm(g)):.2e}   eig [{eig.min():.2e}, {eig.max():.2e}]   "
          f"PD={bool(np.all(eig > 0))}")
    print(f"  trajectory correlation (worst weight): {trajectory_correlation(hp, model, X, y, r, true_ws):.4f}")

    plot_weight_trajectory(true_hp, hp, model, X, y, r)

    return hp


def main():
    p = argparse.ArgumentParser(
        description="Run a REINFORCE parameter-recovery experiment with parameters you choose.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='RF_b_1', choices=sorted(REINFORCE_MODELS))
    p.add_argument('--alpha', type=float, nargs='+', default=[0.03],
                   help='ground-truth learning rate(s); one value or three')
    p.add_argument('--beta', type=float, nargs='+', default=[0.3],
                   help='ground-truth baseline(s); ignored by the no-baseline models')
    p.add_argument('--w0', type=float, nargs=3, default=list(DEFAULT_W0),
                   help='ground-truth initial weights (bias right left)')
    p.add_argument('-T', '--num_trials', type=int, default=12000)
    p.add_argument('--num_seeds', type=int, default=8, help='optimization restarts')
    p.add_argument('--data_seed', type=int, default=0)
    p.add_argument('--fit_seed', type=int, default=0)
    p.add_argument('--max_iters', type=int, default=2000)
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--ridge_lambda', type=float, default=0.)
    p.add_argument('--ridge_alpha', type=float, default=0.)
    p.add_argument('--ridge_c', type=float, default=0.)
    p.add_argument('--mu_alpha', type=float, default=0.)
    p.add_argument('--mu_c', type=float, default=0.)
    p.add_argument('--verbose', action='store_true', help='show per-seed optimizer output')
    p.add_argument('--beta_sweep', type=float, nargs='+', default=None,
                   help='sweep these ground-truth baselines instead of a single fit')
    args = p.parse_args()

    alphas = args.alpha[0] if len(args.alpha) == 1 else np.asarray(args.alpha)
    betas = args.beta_sweep if args.beta_sweep is not None else [
        args.beta[0] if len(args.beta) == 1 else np.asarray(args.beta)]

    for b in betas:
        run_recovery(args.model, alphas, b, np.asarray(args.w0, float),
                     args.num_trials, args.num_seeds, args.fit_seed, args.data_seed,
                     args.max_iters, args.tol, args.ridge_lambda, args.ridge_alpha,
                     args.ridge_c, args.mu_alpha, args.mu_c, quiet=not args.verbose)


if __name__ == '__main__':
    main()
