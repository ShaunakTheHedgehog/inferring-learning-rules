import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
from jax.tree_util import tree_map
import optax
import numpy as np
import pickle as pkl

from utils import generate_full_input_set


def shallow_net(params, inputs):
    W = params["W"]     # (2, 3)
    out = W @ inputs    # (3, )
    return out

def shallow_net_params(key, const_mean=0., const_scale=0.1, stim_mean=0., stim_scale=0.1,
                       beta_mean=9., beta_scale=1., lr_mean=0.003, lr_scale=0.001, nonneg=False):
    skeys = jr.split(key, 6)

    c0 = jnp.abs(jr.normal(skeys[1], (1,))) * const_scale + const_mean
    s0 = jnp.abs(jr.normal(skeys[2], (1,))) * stim_scale + stim_mean

    W = jnp.array([[c0, s0, s0], [c0, s0, s0]]).reshape(2, 3) + 1e-4 * jr.normal(skeys[3], (2, 3))
    if nonneg:
        W = jnp.abs(W)

    beta = jnp.abs(jr.normal(skeys[4], (1,)) * beta_scale + beta_mean)
    beta = beta[0]

    lr = jnp.abs(jr.normal(skeys[5], (1,)) * lr_scale + lr_mean)
    lr = lr[0]

    return {"W" : W}, (beta, lr)


def fit_shallow_model(params, num_trials, inputs, targets, lr, beta, key, loss_type='tutor_executor', nonneg_constraint=True,
                      end_condition=True, zero_contrast_mask=None, num_sessions=None, num_grading_levels=2):
    '''
    Train shallow linear network on one of two RL-based loss functions.
    params              :       model parameters, W1 (diagonal) and W2
    num_trials          :       number of training iterations
    inputs              :       3-dimensional inputs (const, R, L), with shape (num_trials, 3)
    targets             :       1-dimensional outputs (0/R or 1/L), with shape (num_trials,)
    lr                  :       learning rate
    beta                :       inverse temperature parameter for softmax policy
    key                 :       random number generator key
    loss_type           :       string indicating which loss function to use
    nonneg_constraint   :   whether to enforce nonnegativity of W1 and W2
    '''
    assert loss_type in ['tutor_executor', 'single_loss_gd']
    assert inputs.shape[0] == targets.shape[0]
    assert inputs.shape[0] == num_trials
    assert inputs.shape[1] == 3

    num_samples_per_session = None
    session_mask = None
    if num_sessions is not None:
        assert num_trials % num_sessions == 0
        num_samples_per_session = num_trials // num_sessions
        session_mask = zero_contrast_mask[:num_samples_per_session].astype(bool)
    threshold_acc_reached_idx = None
    eff_num_trials = num_trials


    def _softmax_policy(out, key):
        Q_R = out[0]
        Q_L = out[1]
        p_R = 1.0 / (1.0 + jnp.exp(-beta * (Q_R - Q_L)))
        p_L = 1 - p_R
        choice_idx = jr.bernoulli(key, p=p_L)
        choice_idx = jnp.where(choice_idx, 1, 0)

        return p_R, p_L, choice_idx

    @jit
    def _reward(params, x, target, key):
        out = shallow_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)
        is_not_zero_contrast = jnp.minimum(2 * num_grading_levels*jnp.sum(x[1:]), 1.)        # get 0 if zero_contrast, 1 if not

        _, sskey = jr.split(key)
        rew = jnp.where((choice_idx == target), 1, 0) * is_not_zero_contrast + jr.bernoulli(sskey, 0.5) * (1 - is_not_zero_contrast)     # account for zero-contrast trials
        rew = jax.lax.stop_gradient(rew)
        return rew, choice_idx

    @jit
    def _dopamine_vars(params, x, rew, key):
        W = params["W"]               # (2, 3)

        out = shallow_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)

        Q_R_stim = jnp.dot(W[0, 1:], x[1:])
        Q_L_stim = jnp.dot(W[1, 1:], x[1:])
        Q_stim = jnp.where((choice_idx == 0), Q_R_stim, Q_L_stim)

        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))

        outcome_DA = rew - Q_stim
        stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim
        return outcome_DA, stim_DA

    @jit
    def _single_loss_gd(params, x, rew, key):
        out = shallow_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)
        Q_ch = out[choice_idx]
        loss = 0.5 * (rew - Q_ch)**2
        return loss

    @jit
    def _stim_loss(params, x, rew, key):
        W = params["W"]               # (2, 3)

        out = shallow_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)
        Q_stim = jnp.dot(W[choice_idx, 1:], x[1:])
        loss = 0.5 * (rew - Q_stim)**2
        return loss

    @jit
    def _const_loss(params, x, rew, key):
        W = params["W"]               # (2, 3)

        out = shallow_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)
        Q_const = W[choice_idx, 0] * x[0]
        loss = 0.5 * (rew - Q_const)**2
        return loss

    @jit
    def _tutor_executor_W2_loss(params, x, rew, key):
        return _stim_loss(params, x, rew, key) + _const_loss(params, x, rew, key)

    @jit
    def _single_loss_gd_step(params, input, rew, key):
        loss_value, grads = jax.value_and_grad(_single_loss_gd)(params, input, rew, key)
        params["W"] -= lr * grads["W"]

        if nonneg_constraint:
            params["W"] = jnp.maximum(params["W"], 0.0)

        return params, loss_value

    @jit
    def _tutor_executor_step(params, input, rew, key):
        loss_value, grads = jax.value_and_grad(_tutor_executor_W2_loss)(params, input, rew, key)
        params["W"] -= lr * grads["W"]

        if nonneg_constraint:
            params["W"] = jnp.maximum(params["W"], 0.0)

        return params, loss_value

    @jit
    def _bias(params):
        x = jnp.array([1., 0., 0.])
        out = shallow_net(params, x)
        p_R, _, _ = _softmax_policy(out, jr.PRNGKey(1))
        return p_R - 0.5


    @jit
    def _psychometric_data(params): # UPDATE THIS
        # inputs, going from left to right
        xs = generate_full_input_set(num_grading_levels)
        data = []

        for x in xs:
            out = shallow_net(params, x)
            p_R, _, _ = _softmax_policy(out, jr.PRNGKey(1))
            data.append(p_R)

        return data

    @jit
    def _stim_DA_bias(params):
        W = params["W"]               # (2, 3)

        right_x = jnp.array([1., 1., 0.])
        Q_R_stim = jnp.dot(W[0, 1:], right_x[1:])
        Q_L_stim = jnp.dot(W[1, 1:], right_x[1:])
        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))
        right_stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim

        left_x = jnp.array([1., 0., 1.])
        Q_R_stim = jnp.dot(W[0, 1:], left_x[1:])
        Q_L_stim = jnp.dot(W[1, 1:], left_x[1:])
        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))
        left_stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim

        return right_stim_DA - left_stim_DA

    losses = []
    rewards = []
    outcome_DA_signals = []
    stim_DA_signals = []
    biases = []
    stim_DA_biases = []
    outputs = []
    q_vals = []

    rightward_statistics = np.zeros((2, 3))      # an array of how often right was chosen in each of the three input settings (right / graded right, zero contrast, left / graded left)
    partial_rightward_statistics = np.zeros_like(rightward_statistics)

    psychometrics = np.zeros((num_trials, 2 * num_grading_levels + 1))

    for n in range(num_trials):
        key, subkey = jr.split(key, 2)
        input, target = inputs[n], targets[n, 0]

        rew, choice_idx = _reward(params, input, target, subkey)
        rewards.append(rew)
        outputs.append(choice_idx)
        q_vals.append(shallow_net(params, input))

        # at the end of each session, we can calculate accuracy and determine whether the model has reached "expert" status
        if (end_condition and n % num_samples_per_session == num_samples_per_session-1):
            curr_rewards = jnp.asarray(rewards[-num_samples_per_session:])      # access all the rewards from the current session
            nonmasked_rewards = curr_rewards[session_mask]     # access only rewards from non-zero contrast trials during the session

            acc = jnp.mean(nonmasked_rewards).item()

            if (acc >= 0.7 and threshold_acc_reached_idx is None):
                threshold_acc_reached_idx = n
                print(f'Threshold acc reached at iter {n+1}')

        if input[1] > 0:   # right trial
            rightward_statistics[1, 0] += 1
            rightward_statistics[0, 0] += (1 - choice_idx)  # add 1 if right was chosen, else add 0
        elif input[1] == 0 and input[2] == 0:   # zero contrast trial
            rightward_statistics[1, 1] += 1
            rightward_statistics[0, 1] += (1 - choice_idx)
        else:   # left trial
            rightward_statistics[1, 2] += 1
            rightward_statistics[0, 2] += (1 - choice_idx)

        if loss_type == 'single_loss_gd':
            params, loss_value = _single_loss_gd_step(params, input, rew, subkey)
        else:
            params, loss_value = _tutor_executor_step(params, input, rew, subkey)

        outcome_DA, stim_DA = _dopamine_vars(params, input, rew, subkey)
        outcome_DA_signals.append(outcome_DA)
        stim_DA_signals.append(stim_DA)

        biases.append(_bias(params))
        stim_DA_biases.append(_stim_DA_bias(params))

        psychometrics[n] = _psychometric_data(params)

        losses.append(loss_value.item())
        if (n+1) % 250 == 0:
            print(f'step {n+1}, loss: {loss_value.item()}')

        if (threshold_acc_reached_idx is not None and n == threshold_acc_reached_idx + 1500):
            partial_rightward_statistics = rightward_statistics.copy()
            # break

        if threshold_acc_reached_idx is None and n == num_trials - 1:
            partial_rightward_statistics = rightward_statistics.copy()


    if threshold_acc_reached_idx is not None:
        eff_num_trials = threshold_acc_reached_idx + 1500 + 1

    r_slopes = psychometrics[:, -1] - psychometrics[:, num_grading_levels]    # full right - zero contrast
    l_slopes = psychometrics[:, num_grading_levels] - psychometrics[:, 0]    # zero contrast - full left
    slope_diffs = r_slopes - l_slopes

    results_dict = {'params' : params, 'losses' : losses, 'rewards' : rewards, 'outputs' : outputs, 'q_vals' : q_vals,
                    'outcome_DA_signals' : outcome_DA_signals, 'stim_DA_signals' : stim_DA_signals,
                    'biases' : biases, 'stim_DA_biases' : stim_DA_biases,
                    'psychometrics' : psychometrics, 'slope_diffs' : slope_diffs, 'rightward_statistics' : rightward_statistics,
                    'partial_rightward_statistics' : partial_rightward_statistics, 'eff_num_trials' : eff_num_trials}

    return params, losses, results_dict