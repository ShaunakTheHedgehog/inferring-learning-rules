import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
from jax.tree_util import tree_map
import numpy as np
import deep_linear_net
from deep_linear_net import cortico_striatal_net
from shallow_linear_net import shallow_net
from utils import generate_graded_contrast_inputs, calculate_model_agnostic_bias, generate_randomized_multisession_graded_data, stratified_quantile_split_indices, calculate_session_accs, generate_full_input_set


def random_init_params(key, W1_stim_std=0.05, W1_const_std=0.1, W2_const_std=0.05, W2_stim_std=0.05,
                       W1_stim_mean=0., W1_const_mean=1., W2_stim_mean=0., W2_const_mean=0.,
                       lr_scale=0.002, lr_mean=0.005, beta_scale=1., beta_mean=13., 
                       model_type='two-layer', enforce_nonnegativity=False):
    """Generate random two-layer linear network parameters"""

    assert model_type in ['two-layer', 'one-layer']
    
    skeys = jr.split(key, 10)
    
    W1_const = jr.normal(skeys[1], (1,)) * W1_const_std + W1_const_mean

    w0 = jr.normal(skeys[2], (1,)) * W1_stim_std + W1_stim_mean
    W1_stim = jnp.concatenate([w0, w0])

    c0 = jr.normal(skeys[3], (1,)) * W2_const_std + W2_const_mean
    W2_const = jnp.array([c0, c0])
    W2_const = W2_const.reshape(2, 1)

    s0 = jr.normal(skeys[4], (1,)) * W2_stim_std + W2_stim_mean
    W2_stim = s0 * jnp.ones((2, 2))

    beta = jr.normal(skeys[5], (1,)) * beta_scale + beta_mean 
    beta = beta[0]

    # lrs = [W1_lr, W2_const_lr, W2_stim_lr]
    lrs = jnp.abs(jr.normal(skeys[6], (3,)) * lr_scale + lr_mean)

    W1_diag = jnp.concatenate([W1_const, W1_stim]) + 1e-4 * jr.normal(skeys[7], (3,))
    W2 = jnp.concatenate([W2_const, W2_stim], axis=1) + 1e-4 * jr.normal(skeys[8], (2, 3))

    if enforce_nonnegativity:
        W1_diag = jnp.abs(W1_diag)
        W2 = jnp.abs(W2)

    params = {"W1_diag" : W1_diag, "W2" : W2}
    if model_type == 'one-layer':
        params = {"W" : W2}
        lrs = lrs[1:]

    return params, beta, lrs


def fit_metric_GD_net(model_type, params, num_trials, num_sessions, inputs, targets, lrs, beta, zero_contrast_mask, 
                      key, nln=jax.nn.identity, nonneg_constraint=False, end_condition=True, num_grading_levels=4, pathway_based=False):
    '''
    Train deep linear network on one of two RL-based loss functions.
    model_type          :       specifies depth of network ('two-layer' or 'one-layer')
    params              :       model parameters, W1 (diagonal) and W2
    num_trials          :       number of training iterations
    inputs              :       3-dimensional inputs (const, R, L), with shape (num_trials, 3)
    targets             :       1-dimensional outputs (0/R or 1/L), with shape (num_trials,)
    lrs                 :       learning rates for W1_diag, W2_const, and W2_stim, respectively
    beta                :       inverse temperature parameter for softmax policy
    key                 :       random number generator key
    nln                 :       nonlinear activation function (default: identity)
    loss_type           :       string indicating which loss function to use
    nonneg_constraint   :       whether to enforce nonnegativity of W1 and W2
    end_condition       :       whether to stop training 1500 iterations after 70% accuracy is attained
    zero_contrast_mask  :       mask for zero-contrast trials
    num_sessions        :       number of sessions to use for multisession data
    '''
    assert inputs.shape[0] == targets.shape[0]
    assert inputs.shape[0] == num_trials
    assert inputs.shape[1] == 3

    num_samples_per_session = None 
    if num_sessions is None:
        assert end_condition is False 
    else:
        assert num_trials % num_sessions == 0
        num_samples_per_session = num_trials // num_sessions
    threshold_acc_reached_idx = None
    eff_num_trials = num_trials

    lrs_gd = None 
    if model_type == 'two-layer':
        W1_lr, W2_const_lr, W2_stim_lr = lrs 

        lrs_W2 = jnp.array([W2_const_lr, W2_stim_lr, W2_stim_lr]).reshape(1, -1)
        if pathway_based:
            assert W1_lr == 0. 
            W1_lr = lrs_W2.copy().reshape(-1)
        
        lrs_gd = {
            "W1_diag": W1_lr,
            "W2": lrs_W2
        }
    else:
        W_const_lr, W_stim_lr = lrs 

        lrs_W = jnp.array([W_const_lr, W_stim_lr, W_stim_lr]).reshape(1, -1)
        
        lrs_gd = {
            "W": lrs_W
        }
    print(f'Training with learning rates: {lrs_gd}')

    
    @jit
    def _softmax_policy(out, key):
        Q_R = out[0]
        Q_L = out[1]
        p_R = 1.0 / (1.0 + jnp.exp(-beta * (Q_R - Q_L)))
        p_L = 1 - p_R
        choice_idx = jr.bernoulli(key, p=p_L)   # 0 corresponds to 'right', 1 to 'left'
        choice_idx = jnp.where(choice_idx, 1, 0)

        return p_R, p_L, choice_idx
    
    @jit 
    def _run_model(params, x):
        out = None 
        if model_type == 'two-layer':
            out, _ = cortico_striatal_net(params, x, nln=nln)
        else:
            out = shallow_net(params, x)
        return out 
    

    @jit
    def _reward(params, x, target, key):
        out = _run_model(params, x)
        _, _, choice_idx = _softmax_policy(out, key)
        is_not_zero_contrast = jnp.minimum(2*num_grading_levels*jnp.sum(x[1:]), 1.)        # get 0 if zero_contrast, 1 if not

        _, sskey = jr.split(key)
        rew = jnp.where((choice_idx == target), 1, 0) * is_not_zero_contrast + jr.bernoulli(sskey, 0.5) * (1 - is_not_zero_contrast)     # account for zero-contrast trials
        rew = jax.lax.stop_gradient(rew)
        return rew, choice_idx

    @jit
    def _two_layer_dopamine_vars(params, x, rew, key):
        assert model_type == 'two-layer'
        W1_diag = params["W1_diag"]     # (3,)
        W2 = params["W2"]               # (2, 3)

        out = _run_model(params, x)
        _, _, choice_idx = _softmax_policy(out, key)

        Q_R_stim = jnp.dot(W2[0, 1:], nln(W1_diag[1:] * x[1:]))
        Q_L_stim = jnp.dot(W2[1, 1:], nln(W1_diag[1:] * x[1:]))
        Q_stim = jnp.where((choice_idx == 0), Q_R_stim, Q_L_stim)

        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))

        outcome_DA = rew - Q_stim
        stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim
        return outcome_DA, stim_DA
    
    @jit
    def _one_layer_dopamine_vars(params, x, rew, key):
        assert model_type == 'one-layer'
        W = params["W"]               # (2, 3)

        out = shallow_net(params, x)
        _, _, choice_idx = _softmax_policy(out, key)

        Q_R_stim = jnp.dot(W[0, 1:], x[1:])
        Q_L_stim = jnp.dot(W[1, 1:], x[1:])
        Q_stim = jnp.where((choice_idx == 0), Q_R_stim, Q_L_stim)

        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))

        outcome_DA = rew - Q_stim
        stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim
        return outcome_DA, stim_DA

    @jit
    def _single_loss_gd(params, x, rew, key):
        out = _run_model(params, x)
        _, _, choice_idx = _softmax_policy(out, key)
        Q_ch = out[choice_idx]
        loss = 0.5 * (rew - Q_ch)**2
        return loss
    

    @jit
    def _metric_gd_step(params, input, rew, key):
        loss_value, grads = jax.value_and_grad(_single_loss_gd)(params, input, rew, key)
        params = jax.tree_util.tree_map(lambda p, g, lr: p - lr * g, params, grads, lrs_gd)

        if nonneg_constraint:
            params = jax.tree_util.tree_map(lambda x: jnp.maximum(x, 0), params)

        return params, loss_value


    @jit
    def _zero_contrast_bias(params):
        x = jnp.array([1., 0., 0.])
        out = _run_model(params, x)
        p_R, _, _ = _softmax_policy(out, jr.PRNGKey(1))
        return p_R - 0.5

    @jit
    def _psychometric_data(params): # UPDATE THIS
        # inputs, going from left to right
        xs = generate_full_input_set(num_grading_levels)
        data = []

        for x in xs:
            out = _run_model(params, x)
            p_R, _, _ = _softmax_policy(out, jr.PRNGKey(1))
            data.append(p_R)

        return data

    @jit
    def _two_layer_stim_DA_bias(params):
        assert model_type == 'two-layer'
        W1_diag = params["W1_diag"]     # (3,)
        W2 = params["W2"]               # (2, 3)

        right_x = jnp.array([1., 1., 0.])
        Q_R_stim = jnp.dot(W2[0, 1:], nln(W1_diag[1:] * right_x[1:]))
        Q_L_stim = jnp.dot(W2[1, 1:], nln(W1_diag[1:] * right_x[1:]))
        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))
        right_stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim

        left_x = jnp.array([1., 0., 1.])
        Q_R_stim = jnp.dot(W2[0, 1:], nln(W1_diag[1:] * left_x[1:]))
        Q_L_stim = jnp.dot(W2[1, 1:], nln(W1_diag[1:] * left_x[1:]))
        p_R_stim, p_L_stim, _ = _softmax_policy([Q_R_stim, Q_L_stim], jr.PRNGKey(1))
        left_stim_DA = Q_R_stim * p_R_stim + Q_L_stim * p_L_stim

        return right_stim_DA - left_stim_DA
    

    @jit
    def _one_layer_stim_DA_bias(params):
        assert model_type == 'one-layer'
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
    outputs = []
    q_vals = []
    outcome_DA_signals = []
    stim_DA_signals = []
    zero_contrast_biases = []
    stim_DA_biases = []
    params_list = [params]


    rightward_statistics = np.zeros((2, 3))      # an array of how often right was chosen in each of the three input settings (right / graded right, zero contrast, left / graded left)
    partial_rightward_statistics = np.zeros_like(rightward_statistics)


    psychometrics = np.zeros((num_trials, 2 * num_grading_levels + 1))

    for n in range(num_trials):
        key, subkey = jr.split(key, 2)
        input, target = inputs[n], targets[n, 0]

        rew, choice_idx = _reward(params, input, target, subkey)
        rewards.append(rew.item())
        outputs.append(choice_idx)
        q_vals.append(_run_model(params, input))

        # at the end of each session, we can calculate accuracy and determine whether the model has reached "expert" status
        if (end_condition and n % num_samples_per_session == num_samples_per_session-1):
            curr_mask = zero_contrast_mask[n-num_samples_per_session+1:n+1].astype(bool)
            curr_rewards = jnp.asarray(rewards[-num_samples_per_session:])      # access all the rewards from the current session
            nonmasked_rewards = curr_rewards[curr_mask]                         # access only rewards from non-zero contrast trials during the session

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

        params, loss_value = _metric_gd_step(params, input, rew, subkey)
        params_list.append(params)

        outcome_DA, stim_DA = None, None 
        if model_type == 'two-layer':
            outcome_DA, stim_DA = _two_layer_dopamine_vars(params, input, rew, subkey)
            stim_DA_biases.append(_two_layer_stim_DA_bias(params))
        else:
            outcome_DA, stim_DA = _one_layer_dopamine_vars(params, input, rew, subkey)
            stim_DA_biases.append(_one_layer_stim_DA_bias(params))
        
        outcome_DA_signals.append(outcome_DA)
        stim_DA_signals.append(stim_DA)

        zero_contrast_biases.append(_zero_contrast_bias(params))

        psychometrics[n] = _psychometric_data(params)

        losses.append(loss_value.item())
        if (n+1) % 250 == 0:
            print(f'step {n+1}, loss: {loss_value.item()}')

        if (threshold_acc_reached_idx is not None and n == jnp.minimum(threshold_acc_reached_idx + 1500, num_trials - 1)):
            partial_rightward_statistics = rightward_statistics.copy()

        if threshold_acc_reached_idx is None and n == num_trials - 1:
            partial_rightward_statistics = rightward_statistics.copy()


    if threshold_acc_reached_idx is not None:
        eff_num_trials = jnp.minimum(threshold_acc_reached_idx + 1500 + 1, num_trials)
        # psychometrics = psychometrics[:num_trials]

    # avg_right_choices /= num_trials
    r_slopes = psychometrics[:, -1] - psychometrics[:, num_grading_levels]    # full right - zero contrast
    l_slopes = psychometrics[:, num_grading_levels] - psychometrics[:, 0]    # zero contrast - full left
    slope_diffs = r_slopes - l_slopes

    results_dict = {'params' : params, 'losses' : losses, 'rewards' : rewards, 'outputs' : outputs, 'q_vals' : q_vals,
                    'outcome_DA_signals' : outcome_DA_signals, 'stim_DA_signals' : stim_DA_signals,
                    'biases' : zero_contrast_biases, 'stim_DA_biases' : stim_DA_biases,
                    'psychometrics' : psychometrics, 'slope_diffs' : slope_diffs, 'rightward_statistics' : rightward_statistics,
                    'partial_rightward_statistics' : partial_rightward_statistics, 'eff_num_trials' : eff_num_trials,
                    'params_list' : params_list}

    return params, losses, results_dict



def generate_metric_GD_train_test_split(true_alphas, true_beta, true_model_type, 
                              session_info, key, nln=jax.nn.identity, num_train_sets=1, num_test_sets=1, vary_inputs=False, 
                              true_w0=None, do_bias_stratified_split=False, fixed_inputs=None, nonneg_init_weights=False,
                              pathway_based=False, use_saxe_lak_gd_init=False):
    assert true_model_type in ['two-layer', 'one-layer']

    def _show_bias_stats(results_dict, z):
        eff_num_trials = results_dict['eff_num_trials']
        rightward_statistics = results_dict['rightward_statistics']
        partial_rightward_statistics = results_dict['partial_rightward_statistics']

        model_agnostic_bias = calculate_model_agnostic_bias(rightward_statistics, z)
        partial_agnostic_bias = calculate_model_agnostic_bias(partial_rightward_statistics, z)
        print(f'\nModel-agnostic bias up to (full) {num_trials} iterations: {model_agnostic_bias}')
        print(f'\nModel-agnostic bias up to {eff_num_trials} iterations: {partial_agnostic_bias}')
        return partial_agnostic_bias, model_agnostic_bias, rightward_statistics

    skeys = jr.split(key, 10)

    (num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast) = session_info
    num_samples_per_session = 2 * num_grading_levels * num_samples_per_grading + num_zero_contrast
    num_trials = num_sessions * num_samples_per_session
    z = 1.*num_zero_contrast / num_samples_per_session

    w0 = None 
    if true_w0 is None:
        if use_saxe_lak_gd_init:
            w0, _ = deep_linear_net.random_init_params(skeys[1], 3, 3, 2, default_single_loss=True)
        else:
            w0, _, _ = random_init_params(skeys[1], model_type=true_model_type, enforce_nonnegativity=nonneg_init_weights)
    else:
        w0 = true_w0

    sskey = skeys[2]

    train_test_split = []

    full_data_dict = {'inputs' : [], 'outputs' : [], 'rewards' : [], 'q_vals' : [],
                      'pma_biases' : [], 'ma_biases' : [], 'rightward_stats' : []}

    train_test_dict = {'train_inputs' : [], 'train_outputs' : [], 'train_rewards' : [], 'train_q_vals' : [],
                       'test_inputs' : [], 'test_outputs' : [], 'test_rewards' : [], 'test_q_vals' : [],
                       'train_pma_biases' : [], 'train_ma_biases' : [], 
                       'test_pma_biases' : [], 'test_ma_biases' : [],
                       'train_rightward_stats' : [], 'test_rightward_stats' : [],
                       'alphas' : true_alphas, 'beta' : true_beta, 'w0' : w0, 
                       'num_train_sets' : num_train_sets, 'num_test_sets' : num_test_sets}

    inputs, targets, mask = generate_randomized_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, 
                                                                         num_zero_contrast, skeys[3])
    inputs = jnp.array(inputs)
    if fixed_inputs is not None:
        assert vary_inputs == False
        inputs = fixed_inputs
        targets = jnp.argmax(inputs[:, 1:], axis=1)
        targets = targets.reshape(-1, 1)

    for i in range(num_test_sets+num_train_sets):
        _, sskey, ssskey = jr.split(sskey, 3)
        subkeys = jr.split(ssskey, 5)

        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
        # inputs, targets, mask = generate_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, num_zero_contrast, sskey)

        if vary_inputs:
            inputs, targets, mask = generate_randomized_multisession_graded_data(num_sessions, num_grading_levels, num_samples_per_grading, 
                                                                         num_zero_contrast, subkeys[2])
            inputs = jnp.array(inputs)
        else:
            inputs = jnp.copy(inputs)

        _, _, results_dict = fit_metric_GD_net(true_model_type, w0_copy, num_trials=num_trials, num_sessions=num_sessions,
                                               inputs=inputs, targets=targets, lrs=true_alphas, beta=true_beta,
                                               zero_contrast_mask=mask, key=subkeys[3], nln=nln,
                                               nonneg_constraint=False, end_condition=True, num_grading_levels=num_grading_levels, pathway_based=pathway_based)
        
        # inputs = jnp.array(inputs)
        outputs = jnp.array(results_dict['outputs'])
        rewards = jnp.array(results_dict['rewards'])
        q_vals = jnp.array(results_dict['q_vals'])

        pma_bias, ma_bias, rightward_stats = _show_bias_stats(results_dict, z)

        accs = calculate_session_accs(np.array(results_dict['rewards']), num_samples_per_session, mask)
        print(f'Session accuracies: {accs}')


        full_data_dict['inputs'].append(inputs)
        full_data_dict['outputs'].append(outputs)
        full_data_dict['rewards'].append(rewards)
        full_data_dict['q_vals'].append(q_vals)
        full_data_dict['pma_biases'].append(pma_bias)
        full_data_dict['ma_biases'].append(ma_bias)
        full_data_dict['rightward_stats'].append(rightward_stats)
    
    if do_bias_stratified_split:
        all_ma_biases = np.array(full_data_dict['ma_biases'])
        train_idxs, test_idxs = stratified_quantile_split_indices(all_ma_biases, K=num_train_sets, n_bins=3, random_state=np.random.randint(1_000))
    else:
        train_idxs = np.arange(num_train_sets)
        test_idxs = np.arange(num_train_sets, num_train_sets + num_test_sets)
    
    for idx in train_idxs:
        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
        train_test_split.append((full_data_dict['inputs'][idx], full_data_dict['outputs'][idx], full_data_dict['rewards'][idx], w0_copy))
    for idx in test_idxs:
        w0_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), w0)
        train_test_split.append((full_data_dict['inputs'][idx], full_data_dict['outputs'][idx], full_data_dict['rewards'][idx], w0_copy))
    
    train_test_dict['train_inputs'] = [full_data_dict['inputs'][i] for i in train_idxs]
    train_test_dict['train_outputs'] = [full_data_dict['outputs'][i] for i in train_idxs]
    train_test_dict['train_rewards'] = [full_data_dict['rewards'][i] for i in train_idxs]
    train_test_dict['train_q_vals'] = [full_data_dict['q_vals'][i] for i in train_idxs]
    train_test_dict['train_pma_biases'] = [full_data_dict['pma_biases'][i] for i in train_idxs]
    train_test_dict['train_ma_biases'] = [full_data_dict['ma_biases'][i] for i in train_idxs]
    train_test_dict['train_rightward_stats'] = [full_data_dict['rightward_stats'][i] for i in train_idxs]
    
    train_test_dict['test_inputs'] = [full_data_dict['inputs'][i] for i in test_idxs]
    train_test_dict['test_outputs'] = [full_data_dict['outputs'][i] for i in test_idxs]
    train_test_dict['test_rewards'] = [full_data_dict['rewards'][i] for i in test_idxs]
    train_test_dict['test_q_vals'] = [full_data_dict['q_vals'][i] for i in test_idxs]
    train_test_dict['test_pma_biases'] = [full_data_dict['pma_biases'][i] for i in test_idxs]
    train_test_dict['test_ma_biases'] = [full_data_dict['ma_biases'][i] for i in test_idxs]
    train_test_dict['test_rightward_stats'] = [full_data_dict['rightward_stats'][i] for i in test_idxs]

    
    for key in train_test_dict.keys():
        if key in ['train_inputs', 'train_outputs', 'train_rewards', 'train_q_vals', 'test_inputs', 'test_outputs', 'test_rewards', 'test_q_vals']:
            train_test_dict[key] = jnp.asarray(train_test_dict[key])
    
    return train_test_split, train_test_dict



