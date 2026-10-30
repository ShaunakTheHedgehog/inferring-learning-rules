# Inferring learning rules in deep neural network architectures from animal learning data

Code for our NeurIPS 2026 paper.

A major goal in computational neuroscience is to understand how animals learn to perform a new behavior from a sequence of actions and rewards. Previous work on inferring animal learning rules from behavioral data has focused on either 1-layer networks or fits to group behavior. Here, we develop a method for inferring **single-animal, deep neural network-based learning rules**, and use it to ask which architectural components — not only depth but also nonlinearity — account for individual animal learning. This method provides a way to rigorously validate which learning rules and architectures can successfully capture single-animal learning.

## Approach

We model trial-by-trial learning in individual mice trained on a sensory decision-making task with three "minimal" network architectures:

| Model | Output |
|---|---|
| 1-layer linear (1L) | $Q = Wx$ |
| 2-layer linear (2L) | $Q = W^2 W^1 x$ |
| 2-layer nonlinear (2NL) | $Q = W^2\,\sigma(W^1 x)$, with $\sigma = \mathrm{erf}$ |

Each network samples choices through a softmax policy with gain $\beta$, and its weights are updated by gradient descent on the reward-prediction error, $\theta_{t+1} = \theta_t - \alpha \nabla_\theta (r_t - Q_{t,c_t})^2$. Given a single animal's sequence of inputs, choices, and rewards, we solve the *inverse problem*: recover the learning rate $\alpha$, softmax gain $\beta$, and initial weights $\theta_1$ that maximize the (regularized) likelihood of the animal's choices. Our core methodological contribution is to develop a statistical method to solve this inverse problem. 

## Main takeaways

1. **Architecture is not fully identifiable.** We prove that GD in a shallow network can be arbitrarily well approximated by GD in a "lazy" 2-layer network, establishing a theoretical limit on network identifiability. We use this result to motivate adding regularization to our inference objective.
2. **Depth and nonlinearity both matter.** Applying our method, we find that the 2NL model achieves the best fit (by AIC) for 57 out of 58 mice across two datasets. Focusing on one of the datasets (n = 22), we found that the 2NL model achieves the best 5-fold cross validation performance for 20 out of 22 mice. Moreover, re-simulated learning trajectories from a fitted 2NL model closely track the timecourse of bias, small-contrast sensitivity, and accuracy in the respective mouse.
3. **Inferred parameters predict learning outcomes.** Across mice in one dataset (n=22), the inferred learning rate and initial first-layer weights of the 2NL model systematically predict individual differences in accuracy and task bias.
4. **Stochastic choice can drive divergent learning.** Models with identical parameters and inputs can "bifurcate" toward opposite biases, suggesting that stochasticity in decision-making itself may shape individual learning trajectories.

## Repository map

| File | Purpose |
|---|---|
| `inference_pipeline.py` | Simulates GD learning trajectories, computes the choice NLL, and runs multi-seed L-BFGS fits (`run_randomized_optimization_procedure`) |
| `inference_utils.py` | Network forward passes, GD learning-rule updates, softmax log-likelihood |
| `utils.py` | Data handling and behavioral metrics (bias, psychometrics, running accuracy, train/test masks) |
| `globals.py` | Shared constants (model classes, regularization strengths, expert threshold) |
| `synthetic_data_inference_analysis.py` | Parameter recovery and identifiability on synthetic data (Sec. 3) |
| `mouse_inference_data_analysis.py` | Single-mouse analyses: AIC comparisons, re-trained models, PCA of inferred parameters (Sec. 4) |
| `figure_generation.py` | Paper figures |
| `metric_GD_network.py`, `deep_linear_net.py`, `shallow_linear_net.py` | Network simulation helpers |
| `tests/` | Unit and integration tests |
| `archived/` | Legacy scripts, kept for reference |

Additional analyses: K-fold cross-validation (`cross_val_fitting.py`, `cross_val_analysis.py`), REINFORCE-based learning rules (`reinforce_pipeline.py`), and PsyTrack baselines (`witten_psytrack_fitting.py`, `witten_learning_fitting.py`; these require `psytrack_learning` from Ashwood et al., 2020).

## Data

We analyze two mouse datasets, which are not included in this repository:

- **Pan-Vazquez et al. (2024)** (`'witten'` in the code; 22 mice) at `witten_data_GD_analysis_final/witten_mouse_data.pkl`
- **Liebana et al. (2025)** (`'saxe'` in the code; 36 mice) at `saxe_data_GD_analysis_final/saxe_mouse_data.pkl`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

<!-- ```bash
python -m pytest tests/ -m "not slow"   # fast unit tests
python -m pytest tests/                 # includes parameter-recovery tests
``` -->

## Citation

```bibtex
@inproceedings{bhandarkar2026inferring,
  title     = {Inferring learning rules in deep neural network architectures from animal learning data},
  author    = {Bhandarkar, Shaunak and Pillow, Jonathan W.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
