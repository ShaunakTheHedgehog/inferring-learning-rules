import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
from jax.tree_util import tree_map
import optax
import numpy as np
import pickle as pkl


def are_valid_learning_rates(x1, x2, x3, x4, num_blocks):
    '''
    Makes sure that four learning rate parameters are all valid when passed into the fit_deep_net function.
    '''
    vars = [x1, x2, x3, x4]
    
    # Check if all are scalars
    if all(jnp.isscalar(v) for v in vars):
        return all(jnp.asarray(v) >= 0 for v in vars)
    
    # Check if all are arrays
    if all(isinstance(v, jnp.ndarray) for v in vars):
        # Get their shapes
        shapes = [v.shape for v in vars]
        # Ensure all shapes are the same and 1D
        same_shape = all(s == num_blocks for s in shapes)
        all_1d = all(len(s) == 1 for s in shapes)
        all_positive = all(jnp.all(v >= 0) for v in vars)
        return same_shape and all_1d and all_positive
    
    return False


def cortico_striatal_net(params, inputs):
    """
    Simple shallow linear network model from Liebana Garcia et al. (2023)

    Arguments:
    - params:   dictionary storing the network weights
    - inputs:   3-dimensional inputs (const, R, L)
    """

    W1_diag = params["W1_diag"]     # (3,)
    W2 = params["W2"]               # (2, 3)

    hidden = W1_diag * inputs
    out = W2 @ hidden

    return out, hidden

def manual_init_params(W1_diag, W2):
    params = {}
    params["W1_diag"] = W1_diag
    params["W2"] = W2
    return params


def random_init_params(key, input_dim, hidden_dim, output_dim, default_tutor_executor=False, default_single_loss=False,
                        W1_stim_range=0.03, W1_const_range=0.1, W2_const_range=0.03, W2_stim_range=0.03,
                       W1_stim_mean=0.03, W1_const_mean=1., W2_stim_mean=0.03, W2_const_mean=0.03):
    """Generate random shallow linear network parameters"""
    assert not (default_tutor_executor and default_single_loss)

    skeys = jr.split(key, 10)

    W1_const, W1_stim, W2_const, W2_stim = None, None, None, None
    beta, lr = None, None

    if default_tutor_executor:
        W1_const = -1.
        key_1 = skeys[1]
        while W1_const < 0:
            _, key_1 = jr.split(key_1)
            W1_const = jr.normal(key_1, (1,)) * 0.1 + 1.

        w0 = -1.
        key_2 = skeys[2]
        while w0 < 0:
            _, key_2 = jr.split(key_2)
            w0 = jr.normal(key_2, (1,)) * 0.05 + 0.05
        W1_stim = jnp.concatenate([w0, w0])

        c0 = 0.
        # key_3 = skeys[3]
        # while c0 < 0:
        #     _, key_3 = jr.split(key_3)
        #     c0 = jr.normal(key_3, (1,)) * 0.05
        W2_const = jnp.array([c0, c0])
        W2_const = W2_const.reshape(2, 1)

        s0 = -1.
        key_4 = skeys[4]
        while s0 < 0:
            _, key_4 = jr.split(key_4)
            s0 = jr.normal(key_4, (1,)) * 0.05 + 0.2
        W2_stim = s0 * jnp.ones((2, 2))

        beta = jr.normal(skeys[5], (1,)) * 1. + 9.     # (mu = 9, sigma = 1)
        beta = beta[0]

        lr = -1.
        key_6 = skeys[6]
        while lr < 0:
            _, key_6 = jr.split(key_6)
            lr = jr.normal(key_6, (1,)) * 0.001 + 0.0026
        lr = lr[0]


    elif default_single_loss:
        W1_const = -1.
        key_1 = skeys[1]
        while W1_const < 0:
            _, key_1 = jr.split(key_1)
            W1_const = jr.normal(key_1, (1,)) * 0.01 + 1.

        w0 = -1.
        key_2 = skeys[2]
        while w0 < 0:
            _, key_2 = jr.split(key_2)
            w0 = jr.normal(key_2, (1,)) * 0.001 + 0.05
        W1_stim = jnp.concatenate([w0, w0])

        c0 = 0.
        # key_3 = skeys[3]
        # while c0 < 0:
        #     _, key_3 = jr.split(key_3)
        #     c0 = jr.normal(key_3, (1,)) * 0.01
        W2_const = jnp.array([c0, c0])
        W2_const = W2_const.reshape(2, 1)

        s0 = -1.
        key_4 = skeys[4]
        while s0 < 0:
            _, key_4 = jr.split(key_4)
            s0 = jr.normal(key_4, (1,)) * 0.01 + 0.7
        W2_stim = s0 * jnp.ones((2, 2))

        beta = jr.normal(skeys[5], (1,)) * 0.01 + 13.
        beta = beta[0]

        lr = -1.
        key_6 = skeys[6]
        while lr < 0:
            _, key_6 = jr.split(key_6)
            lr = jr.normal(key_6, (1,)) * 0.001 + 0.0015
        lr = lr[0]

    else:
        W1_const = jr.uniform(skeys[1], (1,), minval=-W1_const_range, maxval=W1_const_range) + W1_const_mean
        W1_stim = jr.uniform(skeys[2], (input_dim-1,), minval=-W1_stim_range, maxval=W1_stim_range) + W1_stim_mean

        W2_const = jr.uniform(skeys[2], (output_dim, 1), minval=-W2_const_range, maxval=W2_const_range) + W2_const_mean
        W2_stim = jr.uniform(skeys[3], (output_dim, hidden_dim-1), minval=-W2_stim_range, maxval=W2_stim_range) + W2_stim_mean

    W1_diag = jnp.concatenate([W1_const, W1_stim]) + 1e-4 * jr.normal(skeys[7], (3,))
    W2 = jnp.concatenate([W2_const, W2_stim], axis=1) + 1e-4 * jr.normal(skeys[8], (2, 3))

    W1_diag = jnp.abs(W1_diag)
    W2 = jnp.abs(W2)


    return {"W1_diag" : W1_diag, "W2" : W2}, (beta, lr)


def fit_deep_net(params, num_trials, inputs, targets, lr, beta, key, loss_type='tutor_executor', nonneg_constraint=True,
        end_condition=True, zero_contrast_mask=None, num_sessions=None, num_grading_levels=2, gd_te_lrs=None, num_lr_blocks=1):
    '''
    Train deep linear network on one of two RL-based loss functions.
    params              :       model parameters, W1 (diagonal) and W2
    num_trials          :       number of training iterations
    inputs              :       3-dimensional inputs (const, R, L), with shape (num_trials, 3)
    targets             :       1-dimensional outputs (0/R or 1/L), with shape (num_trials,)
    lr                  :       learning rate
    beta                :       inverse temperature parameter for softmax policy
    key                 :       random number generator key
    loss_type           :       string indicating which loss function to use
    nonneg_constraint   :   whether to enforce nonnegativity of W1 and W2
    end_condition       :   whether to stop training 1500 iterations after 70% accuracy is attained
    zero_contrast_mask  :   mask for zero-contrast trials
    num_sessions        :   number of sessions to use for multisession data
    measure_stats_until :   number of trials during which to measure performance
    '''
    assert loss_type in ['tutor_executor', 'single_loss_gd', 'gd_and_te']
    assert inputs.shape[0] == targets.shape[0]
    assert inputs.shape[0] == num_trials
    assert inputs.shape[1] == 3
    num_samples_per_session = None
    session_mask = None
    if num_sessions is not None:
        assert num_trials % num_sessions == 0
        assert zero_contrast_mask is not None
        num_samples_per_session = num_trials // num_sessions
        session_mask = zero_contrast_mask[:num_samples_per_session].astype(bool)
    threshold_acc_reached_idx = None
    eff_num_trials = num_trials

    gd_W1_lr = lr 
    gd_W2_lr = lr
    te_W2const_lr = lr
    te_W2stim_lr = lr 
    if gd_te_lrs is not None:
        assert lr is None 
        gd_W1_lr, gd_W2_lr, te_W2const_lr, te_W2stim_lr = gd_te_lrs
    if lr is not None:
        assert gd_te_lrs is None 
    
    # assert are_valid_learning_rates(gd_W1_lr, gd_W2_lr, te_W2const_lr, te_W2stim_lr, num_lr_blocks)
    assert num_trials % num_lr_blocks == 0
    block_length = int(num_trials / num_lr_blocks)

    gd_W1_lrs = jnp.repeat(gd_W1_lr, block_length)
    gd_W2_lrs = jnp.repeat(gd_W2_lr, block_length)

    te_W2const_lrs = jnp.repeat(te_W2const_lr, block_length)
    te_W2stim_lrs = jnp.repeat(te_W2stim_lr, block_length)

    te_W2_lrs = jnp.hstack((te_W2const_lrs, te_W2stim_lrs, te_W2stim_lrs))  # jnp.array([te_W2const_lr, te_W2stim_lr, te_W2stim_lr]).reshape(1, -1)

    @jit
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
        out, _ = cortico_striatal_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)
        is_not_zero_contrast = jnp.minimum(2*num_grading_levels*jnp.sum(x[1:]), 1.)        # get 0 if zero_contrast, 1 if not

        _, sskey = jr.split(key)
        rew = jnp.where((choice_idx == target), 1, 0) * is_not_zero_contrast + jr.bernoulli(sskey, 0.5) * (1 - is_not_zero_contrast)     # account for zero-contrast trials
        rew = jax.lax.stop_gradient(rew)
        return rew, choice_idx

    @jit
    def _dopamine_vars(params, x, rew, key):
        W1_diag = params["W1_diag"]     # (3,)
        W2 = params["W2"]               # (2, 3)

        out, hidden = cortico_striatal_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)

        Q_R_stim = jnp.dot(W2[0, 1:], W1_diag[1:] * x[1:])
        Q_L_stim = jnp.dot(W2[1, 1:], W1_diag[1:] * x[1:])
        Q_stim = jnp.where((choice_idx == 0), Q_R_stim, Q_L_stim)

        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))

        outcome_DA = rew - Q_stim
        stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim
        return outcome_DA, stim_DA

    @jit
    def _single_loss_gd(params, x, rew, key):
        out, hidden = cortico_striatal_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)
        Q_ch = out[choice_idx]
        loss = 0.5 * (rew - Q_ch)**2
        return loss

    @jit
    def _stim_loss(params, x, rew, key):
        W1_diag = params["W1_diag"]     # (3,)
        W2 = params["W2"]               # (2, 3)

        out, hidden = cortico_striatal_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)
        Q_stim = jnp.dot(W2[choice_idx, 1:], W1_diag[1:] * x[1:])
        loss = 0.5 * (rew - Q_stim)**2
        return loss

    @jit
    def _const_loss(params, x, rew, key):
        W1_diag = params["W1_diag"]     # (3,)
        W2 = params["W2"]               # (2, 3)

        out, hidden = cortico_striatal_net(params, x)
        p_R, p_L, choice_idx = _softmax_policy(out, key)
        Q_const = W2[choice_idx, 0] * W1_diag[0] * x[0]
        loss = 0.5 * (rew - Q_const)**2
        return loss

    @jit
    def _tutor_executor_W2_loss(params, x, rew, key):
        return _stim_loss(params, x, rew, key) + _const_loss(params, x, rew, key)

    @jit
    def _single_loss_gd_step(params, input, rew, lrs, key):
        curr_gd_W1_lr, curr_gd_W2_lr, _ = lrs
        loss_value, grads = jax.value_and_grad(_single_loss_gd)(params, input, rew, key)
        # params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        params["W1_diag"] -= curr_gd_W1_lr * grads["W1_diag"]
        params["W2"] -= curr_gd_W2_lr * grads["W2"]

        if nonneg_constraint:
            params["W1_diag"] = jnp.maximum(params["W1_diag"], 0.0)
            params["W2"] = jnp.maximum(params["W2"], 0.0)

        return params, loss_value

    @jit
    def _tutor_executor_step(params, input, rew, lrs, key):
        curr_gd_W1_lr, _, curr_te_W2_lrs = lrs
        loss_value_W1, grads_W1 = jax.value_and_grad(_single_loss_gd)(params, input, rew, key)
        loss_value_W2, grads_W2 = jax.value_and_grad(_tutor_executor_W2_loss)(params, input, rew, key)

        params["W1_diag"] -= curr_gd_W1_lr * grads_W1["W1_diag"]
        params["W2"] -= curr_te_W2_lrs * grads_W2["W2"]

        if nonneg_constraint:
            params["W1_diag"] = jnp.maximum(params["W1_diag"], 0.0)
            params["W2"] = jnp.maximum(params["W2"], 0.0)

        return params, loss_value_W1 + loss_value_W2

    @jit 
    def _gd_and_te_step(params, input, rew, lrs, key):
        curr_gd_W1_lr, curr_gd_W2_lr, curr_te_W2_lrs = lrs
        # currently using one lr for W1 and another lr for W2 under GD
        gd_loss_value, gd_grads = jax.value_and_grad(_single_loss_gd)(params, input, rew, key)
        # new_params = jax.tree_util.tree_map(lambda p, g: p - gd_lr * g, params, gd_grads)
        new_params = {}
        new_params["W1_diag"] = params["W1_diag"] - curr_gd_W1_lr * gd_grads["W1_diag"]
        new_params["W2"] = params["W2"] - curr_gd_W2_lr * gd_grads["W2"]

        # currently using one lr for W2_const and another lr for W2_stim under TE (W1 update is just GD update for TE)

        # te_loss_value_W1, te_grads_W1 = jax.value_and_grad(_single_loss_gd)(params, input, rew, key)
        te_loss_value_W2, te_grads_W2 = jax.value_and_grad(_tutor_executor_W2_loss)(params, input, rew, key)

        # new_params["W1_diag"] -= te_W1_lr * te_grads_W1["W1_diag"]
        new_params["W2"] -= curr_te_W2_lrs * te_grads_W2["W2"]
        params = new_params

        if nonneg_constraint:
            params["W1_diag"] = jnp.maximum(params["W1_diag"], 0.0)
            params["W2"] = jnp.maximum(params["W2"], 0.0)

        return params, gd_loss_value + te_loss_value_W2

    @jit
    def _bias(params):
        x = jnp.array([1., 0., 0.])
        out, hidden = cortico_striatal_net(params, x)
        p_R, _, _ = _softmax_policy(out, jr.PRNGKey(1))
        return p_R - 0.5

    @jit
    def _psychometric_data(params): # UPDATE THIS
        # inputs, going from left to right
        grading = 1./num_grading_levels
        gradings = jnp.arange(0, 1. + grading/2, grading)
        R_column = jnp.concatenate([jnp.zeros(num_grading_levels), gradings])
        L_column = jnp.flip(R_column)
        const_column = jnp.ones(2 * num_grading_levels + 1)
        xs = jnp.array([const_column, R_column, L_column]).T
        # xs = None
        # if not add_extra_gradings:
        #     xs = jnp.array([[1., 0., 1.], [1., 0., 0.5],[1., 0., 0.], [1., 0.5, 0.], [1., 1., 0.]])
        # else:
        #     xs = jnp.array([[1., 0., 1.], [1., 0., 0.75], [1., 0., 0.5], [1., 0., 0.25], [1., 0., 0.], [1., 0.25, 0], [1., 0.5, 0.], [1., 0.75, 0.], [1., 1., 0.]])
        data = []

        for x in xs:
            out, hidden = cortico_striatal_net(params, x)
            p_R, _, _ = _softmax_policy(out, jr.PRNGKey(1))
            data.append(p_R)

        return data

    @jit
    def _stim_DA_bias(params):
        W1_diag = params["W1_diag"]     # (3,)
        W2 = params["W2"]               # (2, 3)

        right_x = jnp.array([1., 1., 0.])
        Q_R_stim = jnp.dot(W2[0, 1:], W1_diag[1:] * right_x[1:])
        Q_L_stim = jnp.dot(W2[1, 1:], W1_diag[1:] * right_x[1:])
        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))
        right_stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim

        left_x = jnp.array([1., 0., 1.])
        Q_R_stim = jnp.dot(W2[0, 1:], W1_diag[1:] * left_x[1:])
        Q_L_stim = jnp.dot(W2[1, 1:], W1_diag[1:] * left_x[1:])
        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))
        left_stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim

        return right_stim_DA - left_stim_DA

    losses = []
    rewards = []
    outputs = []
    outcome_DA_signals = []
    stim_DA_signals = []
    biases = []
    stim_DA_biases = []
    params_list = [params]

    # avg_right_choices = 0.

    rightward_statistics = np.zeros((2, 3))      # an array of how often right was chosen in each of the three input settings (right / graded right, zero contrast, left / graded left)
    partial_rightward_statistics = np.zeros_like(rightward_statistics)

    psychometrics = np.zeros((num_trials, 2 * num_grading_levels + 1))

    for n in range(num_trials):
        key, subkey = jr.split(key, 2)
        input, target = inputs[n], targets[n, 0]
        curr_gd_W1_lr = gd_W1_lrs[n]
        curr_gd_W2_lr = gd_W2_lrs[n]
        curr_te_W2_lrs = te_W2_lrs[n]
        curr_lrs = (curr_gd_W1_lr, curr_gd_W2_lr, curr_te_W2_lrs)

        rew, choice_idx = _reward(params, input, target, subkey)
        rewards.append(rew.item())
        outputs.append(choice_idx)
        # avg_right_choices += (1 - choice_idx)

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
            params, loss_value = _single_loss_gd_step(params, input, rew, curr_lrs, subkey)
        elif loss_type == 'tutor_executor':
            params, loss_value = _tutor_executor_step(params, input, rew, curr_lrs, subkey)
        else:
            params, loss_value = _gd_and_te_step(params, input, rew, curr_lrs, subkey)
        params_list.append(params)

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
        # psychometrics = psychometrics[:num_trials]

    # avg_right_choices /= num_trials
    r_slopes = psychometrics[:, -1] - psychometrics[:, num_grading_levels]    # full right - zero contrast
    l_slopes = psychometrics[:, num_grading_levels] - psychometrics[:, 0]    # zero contrast - full left
    slope_diffs = r_slopes - l_slopes

    results_dict = {'params' : params, 'losses' : losses, 'rewards' : rewards, 'outputs' : outputs, 
                    'outcome_DA_signals' : outcome_DA_signals, 'stim_DA_signals' : stim_DA_signals,
                    'biases' : biases, 'stim_DA_biases' : stim_DA_biases,
                    'psychometrics' : psychometrics, 'slope_diffs' : slope_diffs, 'rightward_statistics' : rightward_statistics,
                    'partial_rightward_statistics' : partial_rightward_statistics, 'eff_num_trials' : eff_num_trials,
                    'params_list' : params_list}

    return params, losses, results_dict