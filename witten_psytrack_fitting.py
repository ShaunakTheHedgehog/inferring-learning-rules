"""Fit psytrack to a single mouse from the Witten mouse dataset.

The dataset (`witten_mouse_data.pkl`) is a dict keyed by integer mouse_id. Each
entry holds, among other things:

    'inputs'             (T, 3) float array, columns [bias, right stim, left stim]
    'outputs'            (T,)   float array of choices, 0 = RIGHT, 1 = LEFT
    'targets'            (T,)   float array, the correct side (same coding)
    'trials_per_session' (S,)   int array, trials in each session (sums to T)

CHOICE CODING -- the one thing that must not be gotten wrong
------------------------------------------------------------
Psytrack models p(y_t) = sigmoid(g_t . w_t) as the probability of a RIGHT
choice: `getPosteriorTerms` computes `pR` from the sigmoid and scores it
against `dat['y'] - 1` (psytrack/getMAP.py:309-322). A {0,1}-coded `y` is
shifted to {1,2} on entry to `getMAP` (psytrack/getMAP.py:52-53), so under
{0,1} coding psytrack means 0 = LEFT and 1 = RIGHT. The example notebook
states the same: "Left=1 and Right=2".

The Witten data uses the opposite convention (0 = RIGHT, 1 = LEFT), so choices
are flipped here:  y_psytrack = 1 - outputs.

With the flip, the 'sR' weight comes out positive and 'sL' negative for a mouse
performing the task correctly, since positive right-contrast then pushes
g.w upward, toward the outcome psytrack calls 1 (Right).

WEIGHT ORDERING
---------------
`read_input` iterates over `sorted(weights.keys())` (helper/helperFunctions.py:149),
so the rows of `wMode` are ordered ALPHABETICALLY by weight name, not by dict
insertion order. With names {'bias', 'sL', 'sR'} the row order is:

    wMode[0] = bias,  wMode[1] = sL,  wMode[2] = sR

'sR' and 'sL' are also predefined in psytrack's COLORS/ZORDER dicts, so the
built-in plotting functions work without a custom color map.

Usage
-----
    python witten_psytrack_fitting.py --mouse_id 13
    python witten_psytrack_fitting.py --mouse_id 13 --no_sigday
    python witten_psytrack_fitting.py --mouse_id 13 --plot
    python witten_psytrack_fitting.py --mouse_id 13 --xval --xval_folds 10
"""

import argparse
import os
import pickle

import numpy as np

import psytrack as psy
from psytrack.getMAP import getPosteriorTerms
from psytrack.helper.helperFunctions import read_input

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "witten_mouse_data.pkl"
)

# Weight names. Keep these in sync with the alphabetical-ordering note above.
BIAS, S_RIGHT, S_LEFT = "bias", "sR", "sL"


# ---------------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------------

def load_mouse(mouse_id, data_path=DEFAULT_DATA_PATH):
    """Pull the raw arrays for one mouse out of the pickle.

    Returns (inputs, outputs, trials_per_session, targets), all numpy arrays in
    the dataset's own coding (0 = right, 1 = left). `np.asarray` makes this
    agnostic to whether the pickle holds numpy or jax arrays.
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    if mouse_id not in data:
        raise KeyError(
            f"mouse_id {mouse_id} not in dataset. Available: {sorted(data.keys())}"
        )
    mouse = data[mouse_id]

    inputs = np.asarray(mouse["inputs"], dtype=float)
    outputs = np.asarray(mouse["outputs"], dtype=float)
    rewards = np.asarray(mouse["rewards"], dtype=float)
    targets = 1 - ((outputs.astype(np.int32) + rewards.astype(np.int32)) % 2)
    tps = mouse.get("trials_per_session", None)
    if tps is not None:
        tps = np.asarray(tps, dtype=int)

    # Sanity checks on the assumed structure.
    if inputs.ndim != 2 or inputs.shape[1] != 3:
        raise ValueError(f"expected inputs of shape (T, 3), got {inputs.shape}")
    if outputs.shape != (inputs.shape[0],):
        raise ValueError(
            f"inputs/outputs length mismatch: {inputs.shape} vs {outputs.shape}"
        )
    if not np.array_equal(np.unique(outputs), [0.0, 1.0]):
        raise ValueError(f"expected choices in {{0, 1}}, got {np.unique(outputs)}")
    if not np.all(inputs[:, 0] == 1.0):
        raise ValueError("expected column 0 of inputs to be an all-ones bias column")

    both_nonzero = np.sum((inputs[:, 1] != 0) & (inputs[:, 2] != 0))
    both_zero = np.sum((inputs[:, 1] == 0) & (inputs[:, 2] == 0))
    if both_nonzero or both_zero:
        print(
            f"  NOTE: {both_nonzero} trials with both contrasts nonzero, "
            f"{both_zero} with both zero (model handles these fine)", flush=True
        )

    return inputs, outputs, tps, targets


def normalize_contrast(c, p=5.0):
    """tanh(p*c)/tanh(p) -- the standard IBL contrast normalization.

    Raw contrasts are {0, 0.0625, 0.125, 0.25, 1.0}: heavily bunched near zero,
    with the largest 16x the smallest. Fitting on them forces the stimulus
    weights to grow ~1/contrast to produce any decision signal, and the
    evidence optimizer responds by inflating the corresponding sigma, which
    lets the weight random-walk almost freely and overfits.

    On mouse 13 this transform raised the log-evidence from -4345 to -4144 and
    dropped max|w| from 65.7 to 7.8, so it is worth considering -- but it is
    OFF by default, since it changes the input representation you specified.
    """
    return np.tanh(p * c) / np.tanh(p)


def build_dataset(inputs, outputs, trials_per_session=None, targets=None,
                  name=None, normalize=False, norm_p=5.0):
    """Format one mouse's data as a psytrack dataset dict `D`.

    Applies the choice flip documented at the top of this file. `D['inputs']`
    holds (T, 1) column arrays; only the first `weights[key]` columns of each
    are used, so one column per regressor is all that is needed.

    Args:
        normalize: if True, pass the contrasts through `normalize_contrast`.
    """
    T = outputs.shape[0]

    # THE FLIP: dataset 0=right/1=left  ->  psytrack 0=left/1=right.
    y = (1 - outputs).astype(int)

    D = {
        "y": y,
        "inputs": {
            # 'bias' is a special key: read_input fills this column with 1s and
            # never reads the array below. Stored anyway for transparency.
            BIAS: np.ones((T, 1)),
            S_RIGHT: inputs[:, 1:2].copy(),
            S_LEFT: inputs[:, 2:3].copy(),
        },
    }

    if normalize:
        for key in (S_RIGHT, S_LEFT):
            D["inputs"][key] = normalize_contrast(D["inputs"][key], p=norm_p)

    if trials_per_session is not None:
        D["dayLength"] = np.asarray(trials_per_session, dtype=int)
        if D["dayLength"].sum() != T:
            raise ValueError(
                f"trials_per_session sums to {D['dayLength'].sum()}, expected {T}"
            )
    if targets is not None:
        # Optional key used by plot_performance; flip to match y's coding.
        D["answer"] = (1 - targets).astype(int)
        D["correct"] = (D["y"] == D["answer"]).astype(int)
    if name is not None:
        D["name"] = name

    _verify_coding(D, inputs, outputs)
    return D


def _verify_coding(D, raw_inputs, raw_outputs):
    """Assert the flip landed the right way round.

    In the raw data a right-stimulus trial has target 0 and a left-stimulus
    trial has target 1. After flipping, a right-stimulus trial must have
    answer 1 -- i.e. psytrack's "Right" outcome must line up with the
    right-contrast regressor, otherwise the fitted weights are sign-mirrored.
    """
    y = D["y"]
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError(f"flipped y is not {{0, 1}}: {np.unique(y)}")
    if not np.array_equal(y, 1 - raw_outputs.astype(int)):
        raise ValueError("flip did not apply correctly")

    if "answer" in D:
        right_trials = raw_inputs[:, 1] > 0
        left_trials = raw_inputs[:, 2] > 0
        if right_trials.any() and not np.all(D["answer"][right_trials] == 1):
            raise ValueError(
                "right-contrast trials must have answer == 1 (psytrack 'Right') "
                "after flipping -- choice coding is backwards"
            )
        if left_trials.any() and not np.all(D["answer"][left_trials] == 0):
            raise ValueError(
                "left-contrast trials must have answer == 0 (psytrack 'Left') "
                "after flipping -- choice coding is backwards"
            )


def get_dataset(mouse_id, data_path=DEFAULT_DATA_PATH, normalize=False, norm_p=5.0):
    """Convenience wrapper: load_mouse + build_dataset."""
    inputs, outputs, tps, targets = load_mouse(mouse_id, data_path)
    return build_dataset(inputs, outputs, tps, targets,
                         name=f"mouse_{mouse_id}",
                         normalize=normalize, norm_p=norm_p)


# ---------------------------------------------------------------------------
# In-sample fitting and NLL   <-- the main path
# ---------------------------------------------------------------------------

def fit_mouse(D, use_sigday=True, sigma_init=2**-4.0, sig_init=2**4.0,
              sigday_init=2**-4.0, opt_sigma=True, show_opt=0):
    """Fit psytrack to `D` with `hyperOpt`, following the example notebook.

    Args:
        D: psytrack dataset from build_dataset().
        use_sigday: if True, include and optimize the `sigDay` hyperparameter,
            which governs how much weights may jump between sessions. Requires
            D['dayLength']. If False, session boundaries are ignored.
        sigma_init: initial trial-to-trial sigma (per weight).
        sig_init: sigInit, the variability on the very first trial. A single
            large fixed value lets the data decide where weights start.
        sigday_init: initial between-session sigma.
        opt_sigma: whether `sigma` is optimized (vs. held at sigma_init).
        show_opt: verbosity passed through to hyperOpt.

    Returns:
        (hyp, evd, wMode, hess_info, weights) -- `weights` is returned so the
        caller can reuse the exact same dict for NLL and plotting.
    """
    weights = {BIAS: 1, S_RIGHT: 1, S_LEFT: 1}
    K = int(np.sum([weights[i] for i in weights]))

    hyper = {"sigInit": sig_init, "sigma": [sigma_init] * K}
    optList = ["sigma"] if opt_sigma else []

    if use_sigday:
        if "dayLength" not in D:
            raise ValueError("use_sigday=True requires D['dayLength']")
        hyper["sigDay"] = [sigday_init] * K
        optList = optList + ["sigDay"]
    else:
        hyper["sigDay"] = None

    hyp, evd, wMode, hess_info = psy.hyperOpt(
        D, hyper, weights, optList, showOpt=show_opt
    )
    return hyp, evd, wMode, hess_info, weights


def compute_nll(D, wMode, weights, verify=True):
    """In-sample negative log-likelihood of the choices given the MAP weights.

        NLL = -sum_t log p(y_t | w_t, x_t)

    This reproduces the likelihood arithmetic of `getPosteriorTerms`
    (psytrack/getMAP.py:309-322) directly from the fitted `wMode`, which
    `hyperOpt` returns -- `hyperOpt` does not expose the `llstruct` that holds
    `logli` internally.

    Only the likelihood term is computed here: the log-prior over the weight
    random walk is deliberately excluded, so this is comparable to the NLL of
    a model with no prior over its parameters.

    Args:
        verify: if True, independently recompute logli by calling
            `getPosteriorTerms` and assert the two agree.

    Returns:
        dict with 'nll', 'logli', 'nll_per_trial', 'bits_per_trial',
        'p_right' (per-trial predicted P(Right)), and 'logli_per_trial'.
    """
    g = read_input(D, weights)                    # (N, K), columns alphabetical
    N, K = g.shape
    if wMode.shape != (K, N):
        raise ValueError(f"expected wMode of shape {(K, N)}, got {wMode.shape}")

    # getMAP mutates D['y'] in place from {0,1} to {1,2}, so normalize rather
    # than assuming either coding.
    y_raw = np.asarray(D["y"])
    uniq = np.unique(y_raw)
    if np.array_equal(uniq, [1, 2]):
        y = y_raw - 1
    elif np.array_equal(uniq, [0, 1]):
        y = y_raw.copy()
    else:
        raise ValueError(f"y must be {{0,1}} or {{1,2}}, got {uniq}")

    gw = np.sum(g * wMode.T, axis=1)
    # log p(y) = y*gw - log(1 + exp(gw)); logaddexp is the stable form.
    logli_per_trial = y * gw - np.logaddexp(0, gw)
    logli = float(np.sum(logli_per_trial))

    if verify:
        _verify_logli(D, wMode, weights, logli)

    return {
        "nll": -logli,
        "logli": logli,
        "nll_per_trial": -logli / N,
        "bits_per_trial": -logli / (N * np.log(2)),
        "p_right": 1 / (1 + np.exp(-gw)),
        "logli_per_trial": logli_per_trial,
        "gw": gw,
    }


def _verify_logli(D, wMode, weights, logli, tol=1e-8):
    """Cross-check `compute_nll` against psytrack's own likelihood code.

    Inverts W = cumsum(E, axis=1) (psytrack/getMAP.py:305-306) to recover the
    innovations E_flat that produced `wMode`, then calls getPosteriorTerms and
    compares its `logli` to ours.
    """
    E = np.hstack([wMode[:, :1], np.diff(wMode, axis=1)])
    E_flat = E.flatten(order="C")

    # getPosteriorTerms needs a hyper dict to build the prior; the prior does
    # not enter logli, so any valid sigma works here.
    K = wMode.shape[0]
    dummy_hyper = {"sigma": [2**-4.0] * K, "sigInit": 2**4.0}
    _, liTerms, W = getPosteriorTerms(E_flat, D, dummy_hyper, weights)

    if not np.allclose(W, wMode, atol=1e-10):
        raise AssertionError("failed to reconstruct wMode from its innovations")
    if abs(liTerms["logli"] - logli) > tol * max(1.0, abs(logli)):
        raise AssertionError(
            f"logli mismatch: direct {logli!r} vs getPosteriorTerms "
            f"{liTerms['logli']!r}"
        )


# ---------------------------------------------------------------------------
# Cross-validation   <-- separate, optional, slow; not used by the in-sample path
# ---------------------------------------------------------------------------

def cross_validate_mouse(D, use_sigday=True, folds=10, seed=41,
                         sigma_init=2**-4.0, sig_init=2**4.0,
                         sigday_init=2**-4.0):
    """Held-out NLL via psytrack's `crossValidate`. Refits `folds` models.

    Unlike the in-sample NLL, every trial here is scored by a model fit without
    it, so this is the number directly comparable to a held-out NLL from
    another model. `crossValidate` requires N to be divisible by `folds`, so
    the dataset is trimmed from the front if needed.

    Returns dict with 'xval_nll', 'xval_logli', 'xval_pL', 'n_trials'.
    """
    weights = {BIAS: 1, S_RIGHT: 1, S_LEFT: 1}
    K = int(np.sum([weights[i] for i in weights]))

    hyper = {"sigInit": sig_init, "sigma": [sigma_init] * K}
    optList = ["sigma"]
    if use_sigday:
        if "dayLength" not in D:
            raise ValueError("use_sigday=True requires D['dayLength']")
        hyper["sigDay"] = [sigday_init] * K
        optList = optList + ["sigDay"]
    else:
        hyper["sigDay"] = None

    N = len(D["y"])
    remainder = N % folds
    if remainder:
        # trim(START=...) drops leading trials and keeps dayLength consistent.
        print(f"  trimming {remainder} leading trials so N is divisible by {folds}")
        D = psy.trim(D, START=remainder)

    xval_logli, xval_pL = psy.crossValidate(
        D, hyper, weights, optList, F=folds, seed=seed
    )
    print()

    n = len(D["y"])
    return {
        "xval_nll": -xval_logli,
        "xval_logli": xval_logli,
        "xval_nll_per_trial": -xval_logli / n,
        "xval_pL": xval_pL,
        "n_trials": n,
        "trimmed_D": D,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(mouse_id, data_path=DEFAULT_DATA_PATH, use_sigday=True, plot=False,
        show_opt=0, normalize=False):
    """Full pipeline for one mouse. Returns a dict of everything computed."""
    print(f"=== mouse {mouse_id} ===", flush=True)
    D = get_dataset(mouse_id, data_path, normalize=normalize)
    N = len(D["y"])
    print(f"  trials: {N}   contrasts: {'tanh-normalized' if normalize else 'raw'}", flush=True)
    if "dayLength" in D:
        print(f"  sessions: {len(D['dayLength'])}  (sigDay {'ON' if use_sigday else 'OFF'})", flush=True)
    print(f"  P(choice = Right) = {np.mean(D['y'] == 1):.3f}", flush=True)
    if "correct" in D:
        print(f"  accuracy = {np.mean(D['correct']):.3f}", flush=True)

    print("\nFitting (hyperOpt)...", flush=True)
    hyp, evd, wMode, hess_info, weights = fit_mouse(
        D, use_sigday=use_sigday, show_opt=show_opt
    )

    # Row order is alphabetical: bias, sL, sR.
    labels = sorted(weights.keys())
    print("\nOptimized hyperparameters:", flush=True)
    print(f"  sigma   (log2): "
          f"{dict(zip(labels, np.round(np.log2(hyp['sigma']), 3)))}", flush=True)
    if use_sigday and hyp.get("sigDay") is not None:
        print(f"  sigDay  (log2): "
              f"{dict(zip(labels, np.round(np.log2(hyp['sigDay']), 3)))}", flush=True)
    print(f"\n  >>> LOG-EVIDENCE: {evd:.4f} <<<", flush=True)
    print("\nMean weight over trials:", flush=True)
    for i, lab in enumerate(labels):
        print(f"  {lab:>5}: {wMode[i].mean():+.3f}  "
              f"(start {wMode[i, 0]:+.3f} -> end {wMode[i, -1]:+.3f})", flush=True)

    nll = compute_nll(D, wMode, weights, verify=True)
    print("\n--- IN-SAMPLE NLL (verified against getPosteriorTerms) ---", flush=True)
    print(f"  NLL total       : {nll['nll']:.4f}", flush=True)
    print(f"  NLL per trial   : {nll['nll_per_trial']:.6f}", flush=True)
    print(f"  bits per trial  : {nll['bits_per_trial']:.6f}", flush=True)
    print(f"  (chance NLL/trial = {np.log(2):.6f})", flush=True)

    results = {
        "D": D, "hyp": hyp, "evd": evd, "wMode": wMode,
        "hess_info": hess_info, "weights": weights, "nll": nll,
    }


    if plot:
        from matplotlib import pyplot as plt

        psy.plot_weights(
            wMode, weights,
            days=D.get("dayLength"), errorbar=hess_info.get("W_std"),
        )
        plt.title(f"mouse {mouse_id}")
        psy.plot_performance(D)
        psy.plot_bias(D)
        plt.show()

    return results


def list_mouse_ids(data_path=DEFAULT_DATA_PATH):
    """Sorted mouse ids in the dataset, so a SLURM array index can select one.

    The ids are NOT contiguous (13..43, then 901..914), so a 0-based array index
    must be mapped through this list rather than used as a mouse_id directly.
    """
    with open(data_path, "rb") as f:
        return sorted(pickle.load(f).keys())


def save_results(results, mouse_id, out_dir, use_sigday, normalize):
    """Save the fit for one mouse. `evd` (log-evidence) is the headline number.

    Written to a temp file then renamed, so a job killed mid-write cannot leave a
    half-written pickle behind.
    """
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "mouse_id": mouse_id,
        "log_evidence": results["evd"],      # <-- the quantity of interest
        "hyp": results["hyp"],               # optimized sigma (+ sigDay if fitted)
        "wMode": results["wMode"],           # (K, N) MAP weights: bias, sL, sR
        "hess_info": results["hess_info"],
        "weights": results["weights"],
        # in-sample NLL, secondary
        "point_nll": results["nll"]["nll"],
        "point_nll_per_trial": results["nll"]["nll_per_trial"],
        "n_trials": len(results["D"]["y"]),
        # provenance
        "use_sigday": use_sigday,
        "normalize": normalize,
    }
    out_path = os.path.join(out_dir, f"mouse_{mouse_id}.pkl")
    tmp_path = f"{out_path}.tmp{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(record, f)
    os.replace(tmp_path, out_path)
    print(f"\nsaved -> {out_path}", flush=True)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mouse_id", type=int, default=None,
                   help="mouse_id key in the pickle, e.g. 13 (omit if using --task_id)")
    p.add_argument("--task_id", type=int, default=None,
                   help="SLURM array index; selects the Nth mouse from the sorted id "
                        "list (ids are non-contiguous, so this is NOT a mouse_id)")
    p.add_argument("--out_dir", default=None,
                   help="if set, save the fit to <out_dir>/mouse_<id>.pkl")
    p.add_argument("--list_mice", action="store_true",
                   help="print the number of mice (array size) and their ids, then exit")
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--no_sigday", action="store_true",
                   help="ignore session boundaries (do not fit sigDay)")
    # p.add_argument("--normalize", action="store_true",
    #                help="apply tanh(5c)/tanh(5) contrast normalization "
    #                     "(higher evidence, far saner weight scale)")
    p.add_argument("--plot", action="store_true",
                   help="show weight / performance / bias plots")
    p.add_argument("--show_opt", type=int, default=0,
                   help="hyperOpt verbosity (0 quiet, 1 verbose)")
    args = p.parse_args()

    mouse_ids = list_mouse_ids(args.data_path)
    if args.list_mice:
        print(f"{len(mouse_ids)} mice -> use --array=0-{len(mouse_ids) - 1}")
        print("ids:", mouse_ids)
        return

    # Resolve which mouse this invocation handles. --task_id indexes the sorted id
    # list; --mouse_id names a mouse directly.
    if args.task_id is not None:
        if not (0 <= args.task_id < len(mouse_ids)):
            raise SystemExit(f"--task_id {args.task_id} out of range "
                             f"0..{len(mouse_ids) - 1}")
        mouse_id = mouse_ids[args.task_id]
    elif args.mouse_id is not None:
        mouse_id = args.mouse_id
    else:
        raise SystemExit("provide either --mouse_id or --task_id")

    results = run(
        mouse_id,
        data_path=args.data_path,
        use_sigday=not args.no_sigday,
        plot=args.plot,
        show_opt=args.show_opt,
    )

    if args.out_dir is not None:
        save_results(results, mouse_id, args.out_dir,
                     use_sigday=not args.no_sigday, normalize=False)


if __name__ == "__main__":
    main()
