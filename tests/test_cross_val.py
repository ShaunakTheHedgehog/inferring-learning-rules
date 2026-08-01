"""Unit tests for the K-fold cross-validation machinery:
`utils.make_kfold_train_masks` and the `cross_val_fitting.py` driver.

Run with:  .venv/bin/python -m pytest tests/test_cross_val.py -v
"""
import math
import os
import pickle as pkl
import subprocess
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import pytest

from utils import get_nonlinearity, make_kfold_train_masks
from inference_utils import convert_weights_to_dict, train_gd_network
from metric_GD_network import generate_randomized_multisession_graded_data
from inference_pipeline import (
    run_single_learning_forward_trajectory,
    run_train_test_single_learning_forward_trajectory,
)
from reinforce_pipeline import (
    NUM_WEIGHTS, REINFORCE_MODELS, get_num_params,
    run_single_RF_learning_forward_trajectory,
    run_train_test_single_RF_forward_trajectory,
)
import cross_val_fitting as cvf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# --------------------------------------------------------------------------- #
# make_kfold_train_masks
# --------------------------------------------------------------------------- #
class TestKFoldMasks:
    # (num_trials, K, block_length) -- includes partial final blocks and the
    # minimum legal size (num_blocks == K)
    CASES = [(13_762, 5, 20), (10_000, 5, 20), (10_037, 5, 20),
             (100, 5, 20), (2_431, 3, 20), (7_777, 10, 20)]

    @pytest.mark.parametrize("num_trials,K,block_length", CASES)
    def test_shapes_and_count(self, num_trials, K, block_length):
        masks = make_kfold_train_masks(num_trials, jr.PRNGKey(0), K=K, block_length=block_length)
        assert len(masks) == K
        assert all(m.shape == (num_trials,) for m in masks)
        assert all(m.dtype == bool for m in masks)

    @pytest.mark.parametrize("num_trials,K,block_length", CASES)
    def test_folds_are_disjoint_and_exhaustive(self, num_trials, K, block_length):
        """Every trial must be held out in exactly one fold."""
        masks = make_kfold_train_masks(num_trials, jr.PRNGKey(0), K=K, block_length=block_length)
        held_out_count = np.sum(np.stack([~np.asarray(m) for m in masks]), axis=0)
        assert np.all(held_out_count == 1)

    @pytest.mark.parametrize("num_trials,K,block_length", CASES)
    def test_fold_sizes_within_one_block(self, num_trials, K, block_length):
        masks = make_kfold_train_masks(num_trials, jr.PRNGKey(0), K=K, block_length=block_length)
        counts = [int(np.sum(~np.asarray(m))) for m in masks]
        assert sum(counts) == num_trials
        assert all(c > 0 for c in counts), "a fold with no held-out trials would score NLL 0"
        assert (max(counts) - min(counts)) <= block_length

    @pytest.mark.parametrize("num_trials,K,block_length", CASES)
    def test_held_out_trials_are_whole_blocks(self, num_trials, K, block_length):
        """Held-out trials must be aligned to block boundaries, never split mid-block."""
        masks = make_kfold_train_masks(num_trials, jr.PRNGKey(0), K=K, block_length=block_length)
        trial_block = np.arange(num_trials) // block_length
        for m in masks:
            test = ~np.asarray(m)
            for b in range(math.ceil(num_trials / block_length)):
                sel = test[trial_block == b]
                assert sel.all() or (~sel).all()

    def test_same_key_gives_same_folds(self):
        """Identical folds across model classes -> paired model comparison."""
        a = make_kfold_train_masks(13_762, jr.PRNGKey(7))
        b = make_kfold_train_masks(13_762, jr.PRNGKey(7))
        assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))

    def test_different_key_gives_different_folds(self):
        a = make_kfold_train_masks(13_762, jr.PRNGKey(7))
        c = make_kfold_train_masks(13_762, jr.PRNGKey(8))
        assert not all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, c))

    def test_asserts_when_fewer_blocks_than_folds(self):
        # 50 trials / 20 = 3 blocks < K=5, which would leave empty folds
        with pytest.raises(AssertionError):
            make_kfold_train_masks(50, jr.PRNGKey(0), K=5, block_length=20)

    def test_returns_jax_arrays(self):
        masks = make_kfold_train_masks(1_000, jr.PRNGKey(0))
        assert all(isinstance(m, jnp.ndarray) for m in masks)


# --------------------------------------------------------------------------- #
# Folds partition the likelihood (ties the masks to the pipeline)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_animal():
    nln_name = "erf"
    k_data, k_train = jr.split(jr.PRNGKey(0), 2)
    inputs, targets, _ = generate_randomized_multisession_graded_data(5, 4, 15, 6, k_data)
    w0_arr = jnp.array([0.5, 0.3, 0.1, 0.2, -0.1, 0.05, -0.3, 0.4, -0.2])
    _, _, rd = train_gd_network(nln_name, convert_weights_to_dict(w0_arr), inputs, targets,
                                0.01, 6.0, k_train, num_grading_levels=4)
    return dict(nln=get_nonlinearity(nln_name), inputs=inputs,
                outputs=jnp.array(rd["outputs"]), rewards=jnp.array(rd["rewards"]),
                hp=jnp.concatenate((jnp.log(jnp.array([0.01, 6.0])), w0_arr)),
                T=int(jnp.array(rd["outputs"]).shape[0]))


def test_kfold_test_nlls_sum_to_full_data_nll(synthetic_animal):
    """Because folds are disjoint and exhaustive, the K held-out NLLs must sum to
    the full-data NLL -- so CV NLL/trial is simply sum(test_nlls) / num_trials."""
    sa = synthetic_animal
    masks = make_kfold_train_masks(sa["T"], jr.PRNGKey(42), K=5, block_length=20)
    total, _, _ = run_single_learning_forward_trajectory(
        sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], sa["nln"], "gd", None)
    test_sum = 0.0
    for m in masks:
        _, te, _, _ = run_train_test_single_learning_forward_trajectory(
            sa["hp"], sa["inputs"], sa["outputs"], sa["rewards"], m, sa["nln"], "gd", None)
        test_sum += float(te)
    assert np.isclose(test_sum, float(total), atol=1e-6)


# --------------------------------------------------------------------------- #
# The same fold properties, for the REINFORCE models
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rf_animal():
    """A short deterministic-REINFORCE rollout with the real task's input structure
    (exactly one contrast channel active per trial, choices coded 1 = right)."""
    rng = np.random.default_rng(0)
    T = 1500
    side = rng.integers(0, 2, T)
    mag = rng.choice(np.array([0.0625, 0.125, 0.25, 1.0]), T)
    X = np.column_stack([np.ones(T), np.where(side == 1, mag, 0.0), np.where(side == 0, mag, 0.0)])
    alpha, c, w0 = 0.03, 0.009, np.array([0.15, -0.06, -0.08])

    w = w0.copy()
    y, r = np.zeros(T, dtype=int), np.zeros(T)
    for t in range(T):
        p = 1.0 / (1.0 + np.exp(-X[t] @ w))
        y[t] = int(rng.random() < p)
        r[t] = float(y[t] == side[t])
        w = w + (y[t] - p) * (alpha * r[t] - c) * X[t]

    return dict(inputs=jnp.asarray(X), outputs=jnp.asarray(y), rewards=jnp.asarray(r),
                T=T, alpha=alpha, c=c, w0=w0)


def _rf_hp(model, alpha, c, w0):
    """Pack RF hyperparameters in the pipeline layout: [log_alphas, intercepts, w0]."""
    num_alphas, num_intercepts = REINFORCE_MODELS[model]
    return jnp.concatenate([jnp.full(num_alphas, np.log(alpha)),
                            jnp.full(num_intercepts, c),
                            jnp.asarray(w0, dtype=float)])


class TestReinforceFolds:
    """The RF models share cross_val_fitting's masking, so they must satisfy the same
    invariants the NN models do -- in particular that the mask splits only the
    LIKELIHOOD while the weight trajectory still runs over every trial."""

    @pytest.mark.parametrize("model", sorted(REINFORCE_MODELS))
    def test_kfold_test_nlls_sum_to_full_data_nll(self, rf_animal, model):
        """Disjoint + exhaustive folds => the K held-out NLLs sum to the full-data NLL,
        so CV NLL/trial is simply sum(test_nlls) / num_trials."""
        sa = rf_animal
        hp = _rf_hp(model, sa["alpha"], sa["c"], sa["w0"])
        masks = make_kfold_train_masks(sa["T"], jr.PRNGKey(42), K=5, block_length=20)
        total, _, _ = run_single_RF_learning_forward_trajectory(
            hp, model, sa["inputs"], sa["outputs"], sa["rewards"])
        test_sum = sum(
            float(run_train_test_single_RF_forward_trajectory(
                hp, model, sa["inputs"], sa["outputs"], sa["rewards"], m)[1])
            for m in masks)
        assert np.isclose(test_sum, float(total), atol=1e-6)

    @pytest.mark.parametrize("model", sorted(REINFORCE_MODELS))
    def test_train_plus_test_equals_total(self, rf_animal, model):
        sa = rf_animal
        hp = _rf_hp(model, sa["alpha"], sa["c"], sa["w0"])
        mask = make_kfold_train_masks(sa["T"], jr.PRNGKey(7), K=5, block_length=20)[0]
        total, _, _ = run_single_RF_learning_forward_trajectory(
            hp, model, sa["inputs"], sa["outputs"], sa["rewards"])
        tr, te, _, _ = run_train_test_single_RF_forward_trajectory(
            hp, model, sa["inputs"], sa["outputs"], sa["rewards"], mask)
        assert np.isclose(float(tr) + float(te), float(total), atol=1e-6)

    @pytest.mark.parametrize("model", sorted(REINFORCE_MODELS))
    def test_mask_does_not_change_the_weight_trajectory(self, rf_animal, model):
        """The animal genuinely experienced every trial, so masking must not alter the
        state the model is scored from -- only which trials enter the likelihood."""
        sa = rf_animal
        hp = _rf_hp(model, sa["alpha"], sa["c"], sa["w0"])
        mask = make_kfold_train_masks(sa["T"], jr.PRNGKey(7), K=5, block_length=20)[0]
        _, ws_full, lls_full = run_single_RF_learning_forward_trajectory(
            hp, model, sa["inputs"], sa["outputs"], sa["rewards"])
        _, _, ws_masked, lls_masked = run_train_test_single_RF_forward_trajectory(
            hp, model, sa["inputs"], sa["outputs"], sa["rewards"], mask)
        assert np.allclose(np.asarray(ws_full), np.asarray(ws_masked), atol=1e-12)
        assert np.allclose(np.asarray(lls_full), np.asarray(lls_masked), atol=1e-12)

    @pytest.mark.parametrize("model", sorted(REINFORCE_MODELS))
    def test_held_out_nll_is_finite_and_matches_param_count(self, rf_animal, model):
        sa = rf_animal
        hp = _rf_hp(model, sa["alpha"], sa["c"], sa["w0"])
        assert hp.shape == (get_num_params(model),)
        mask = make_kfold_train_masks(sa["T"], jr.PRNGKey(7), K=5, block_length=20)[0]
        tr, te, _, _ = run_train_test_single_RF_forward_trajectory(
            hp, model, sa["inputs"], sa["outputs"], sa["rewards"], mask)
        assert np.isfinite(float(tr)) and np.isfinite(float(te)) and float(te) > 0.0


# --------------------------------------------------------------------------- #
# cross_val_fitting driver
# --------------------------------------------------------------------------- #
class TestCrossValDriver:
    def test_model_classes_match_globals(self):
        from globals import MODEL_CLASS_INFO
        specs = {(v[0], v[1], v[2]) for v in MODEL_CLASS_INFO.values()}
        assert set(cvf.MODEL_CLASSES.values()) == specs

    def test_build_task_list_is_fold_major(self):
        """Fold-major ordering: all mice's fold 0 first, then all mice's fold 1, ...
        so an early-terminated run still yields one train/test fit per mouse."""
        mice = [13, 14, 15]
        tasks = cvf.build_task_list(mice, K=5)
        assert len(tasks) == 15
        # first block of len(mice) tasks is fold 0 for every mouse, in order
        assert tasks[0] == (13, 0) and tasks[1] == (14, 0) and tasks[2] == (15, 0)
        # next block is fold 1 for every mouse
        assert tasks[3] == (13, 1) and tasks[5] == (15, 1)
        # every (mouse, fold) appears exactly once
        assert len(set(tasks)) == len(tasks)

    def test_each_fold_occupies_contiguous_index_range(self):
        """Lets a single fold be submitted alone, e.g. --array=0-34 for fold 0."""
        mice = list(range(35))
        K = 5
        tasks = cvf.build_task_list(mice, K)
        for k in range(K):
            block = tasks[k * len(mice):(k + 1) * len(mice)]
            assert {f for _, f in block} == {k}
            assert [m for m, _ in block] == mice

    def test_first_num_mice_tasks_cover_every_mouse_once(self):
        """The key property: the first wave of tasks gives one fit per mouse."""
        mice = list(range(35))
        tasks = cvf.build_task_list(mice, K=5)
        first_wave = tasks[:len(mice)]
        assert {m for m, _ in first_wave} == set(mice)
        assert all(f == 0 for _, f in first_wave)

    def test_task_ids_cover_all_mouse_fold_pairs(self):
        mice, K = list(range(35)), 5
        tasks = cvf.build_task_list(mice, K)
        assert len(tasks) == 175
        assert {m for m, _ in tasks} == set(mice)
        assert {f for _, f in tasks} == set(range(K))

    def test_get_mouse_arrays_casts_outputs_to_int(self):
        """outputs index into the model's Q-values, so they must be integer-typed."""
        mouse = dict(inputs=np.random.rand(50, 3),
                     outputs=np.random.randint(0, 2, 50).astype(np.float64),
                     rewards=np.random.randint(0, 2, 50).astype(np.float64))
        inputs, outputs, rewards = cvf.get_mouse_arrays(mouse)
        assert jnp.issubdtype(outputs.dtype, jnp.integer)
        assert inputs.shape == (50, 3) and rewards.shape == (50,)

    def test_check_writable_passes_on_writable_dir(self, tmp_path):
        cvf.check_writable(str(tmp_path))  # must not raise

    def test_check_writable_raises_on_readonly_dir(self, tmp_path):
        os.chmod(tmp_path, 0o555)
        try:
            with pytest.raises(SystemExit):
                cvf.check_writable(str(tmp_path))
        finally:
            os.chmod(tmp_path, 0o755)  # restore so pytest can clean up tmp_path

    def test_to_numpy_strips_jax_arrays_from_nested_containers(self):
        nested = {'a': jnp.array([1.0, 2.0]),
                  'b': [jnp.array([3]), {'c': jnp.array(4.0)}],
                  'd': None, 'e': 5}
        converted = cvf.to_numpy(nested)

        def assert_no_jax(x):
            assert not type(x).__module__.startswith('jax'), f'found jax type: {type(x)}'
            if isinstance(x, dict):
                for v in x.values():
                    assert_no_jax(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    assert_no_jax(v)
        assert_no_jax(converted)
        assert isinstance(converted['a'], np.ndarray)
        assert np.array_equal(converted['a'], [1.0, 2.0])
        assert converted['d'] is None and converted['e'] == 5


# --------------------------------------------------------------------------- #
# Paired folds: every model must be scored on the SAME split
# --------------------------------------------------------------------------- #
class TestFoldsArePairedAcrossModels:
    """The NN and RF families share this driver, so the fold partition must depend
    only on (fold_seed, mouse_id) -- never on the model. If it ever did depend on the
    model, NN-vs-RF test NLLs would be computed on different held-out trials and the
    model comparison would be silently unpaired."""

    def test_mask_is_reproducible_from_fold_seed_and_mouse_id_alone(self):
        T, K, bl = 12_167, 5, 20
        a = make_kfold_train_masks(T, jr.fold_in(jr.PRNGKey(0), 13), K=K, block_length=bl)
        b = make_kfold_train_masks(T, jr.fold_in(jr.PRNGKey(0), 13), K=K, block_length=bl)
        assert all(bool(jnp.all(x == y)) for x, y in zip(a, b))

    @pytest.mark.parametrize("seed,mouse", [(1, 13), (0, 14)])
    def test_mask_changes_with_fold_seed_or_mouse(self, seed, mouse):
        T, K, bl = 12_167, 5, 20
        ref = make_kfold_train_masks(T, jr.fold_in(jr.PRNGKey(0), 13), K=K, block_length=bl)
        other = make_kfold_train_masks(T, jr.fold_in(jr.PRNGKey(seed), mouse), K=K, block_length=bl)
        assert not bool(jnp.all(ref[0] == other[0]))

    def test_model_family_classification(self):
        assert [cvf.model_family(m) for m in ['1L', '2L', '2NL']] == ['NN'] * 3
        assert [cvf.model_family(m) for m in ['RF_1', 'RF_b_1', 'RF', 'RF_b']] == ['RF'] * 4


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO_ROOT, 'witten_data_GD_analysis_final',
                                    'witten_mouse_data.pkl')),
    reason="witten mouse data not available")
@pytest.mark.parametrize("task_id", [0, 174])
def test_cli_gives_identical_split_across_families(tmp_path, task_id):
    """One NN model and one RF model, same --task_id: identical trials held out.

    Also pins the deliberate difference: RF fits see the choice-flipped outputs
    (dataset 0=right vs the pipeline's 1=right), which is a relabeling of the same
    trials and so cannot move the split.
    """
    recs = {}
    for model in ['1L', 'RF_b_1']:
        out_dir = str(tmp_path / model)
        res = subprocess.run(
            [sys.executable, "cross_val_fitting.py", "--model", model,
             "--task_id", str(task_id), "--dry_run", "--out_dir", out_dir],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert res.returncode == 0, f"{model} failed:\n{res.stdout[-2000:]}{res.stderr[-2000:]}"
        paths = [os.path.join(dp, f) for dp, _, fs in os.walk(out_dir)
                 for f in fs if f.endswith('.pkl')]
        assert len(paths) == 1, paths
        with open(paths[0], 'rb') as f:
            recs[model] = pkl.load(f)

    nn, rf = recs['1L'], recs['RF_b_1']
    assert np.array_equal(nn['train_mask'], rf['train_mask']), "held-out trials differ"
    for k in ['mouse_id', 'fold', 'K', 'block_length', 'fold_seed',
              'num_trials', 'n_train', 'n_test']:
        assert nn[k] == rf[k], f"{k} differs: {nn[k]} vs {rf[k]}"
    # the one intended difference between the families
    assert nn['family'] == 'NN' and rf['family'] == 'RF'
    assert nn['flipped_outputs'] is False and rf['flipped_outputs'] is True


# --------------------------------------------------------------------------- #
# End-to-end smoke test of the CLI on real data (fast: --dry_run)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO_ROOT, 'witten_data_GD_analysis_final',
                                    'witten_mouse_data.pkl')),
    reason="witten mouse data not available")
@pytest.mark.parametrize("model", ['1L'] + sorted(REINFORCE_MODELS))
def test_cli_dry_run_end_to_end(tmp_path, model):
    """Runs the actual command a SLURM task would run, and checks the saved record.

    Covers one NN model and every REINFORCE model, so the RF branch of the driver
    (optimizer dispatch, choice flip, held-out scoring, saving) is exercised too.
    """
    out_dir = str(tmp_path / "cv_out")
    res = subprocess.run(
        [sys.executable, "cross_val_fitting.py", "--model", model,
         "--task_id", "0", "--dry_run", "--out_dir", out_dir],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}"

    paths = [os.path.join(dp, f) for dp, _, fs in os.walk(out_dir)
             for f in fs if f.endswith('.pkl')]
    assert len(paths) == 1, f"expected one saved fold, got {paths}"

    with open(paths[0], 'rb') as f:
        rec = pkl.load(f)

    # identity + fold metadata sufficient to reproduce the split
    for k in ['dataset', 'mouse_id', 'model', 'family', 'flipped_outputs', 'fold', 'K',
              'block_length', 'fold_seed', 'train_mask', 'num_trials', 'n_train',
              'n_test', 'optim_dict', 'best_hyperparams_arr', 'train_nll', 'test_nll',
              'converged']:
        assert k in rec, f"missing key {k}"

    assert rec['model'] == model
    if model in REINFORCE_MODELS:
        # RF-specific: parameter count, the choice flip, and a flat (3,) w0
        assert rec['family'] == 'RF' and rec['flipped_outputs'] is True
        assert np.asarray(rec['best_hyperparams_arr']).shape == (get_num_params(model),)
        assert np.asarray(rec['best_w0']).shape == (NUM_WEIGHTS,)
    else:
        assert rec['family'] == 'NN' and rec['flipped_outputs'] is False

    assert rec['n_train'] + rec['n_test'] == rec['num_trials']
    assert rec['train_mask'].shape == (rec['num_trials'],)
    assert int((~rec['train_mask']).sum()) == rec['n_test']
    # held-out fraction should be about 1/K
    assert abs(rec['n_test'] / rec['num_trials'] - 1.0 / rec['K']) < 0.02
    assert np.isfinite(rec['test_nll'])


def test_cli_rejects_out_of_range_task_id():
    res = subprocess.run(
        [sys.executable, "cross_val_fitting.py", "--model", "1L",
         "--task_id", "99999"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode != 0
    assert "out of range" in (res.stdout + res.stderr)
