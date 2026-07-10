import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax.tree_util import tree_map
import optax
import optax.tree_utils as otu
from functools import partial
import numpy as np

import matplotlib.pyplot as plt

from utils import flatten_params, compute_two_layer_flattened_net_params, generate_full_input_set, get_nonlinearity
from deep_linear_net import cortico_striatal_net
from shallow_linear_net import shallow_net
from globals import FULL_CONTRAST_LEFT_INPUT, FULL_CONTRAST_RIGHT_INPUT, NUM_TWO_LAYER_WEIGHTS, W1_NUM_WEIGHTS, W2_NUM_WEIGHTS

ONE_LAYER_MAX_PARAMS = 10
NUM_ONE_LAYER_WEIGHTS = 6

def convert_to_dict(hyperparams, inference_model):
    assert inference_model in ['gd', 'full_metric_gd', 'pathway_metric_gd', 'fixed_metric_gd']
    
    depth = get_depth(hyperparams, inference_model)
    
    hyperparams_dict = {}

    if inference_model == 'gd':
        if depth == 'two-layer':
            hyperparams_dict['log_alpha'] = hyperparams[0]
            hyperparams_dict['log_beta'] = hyperparams[1]
            hyperparams_dict['W1_diag_0'] = hyperparams[2:5]
            hyperparams_dict['W2_0'] = hyperparams[5:11].reshape(2, 3)
        else:
            hyperparams_dict['log_alpha'] = hyperparams[0]
            hyperparams_dict['log_beta'] = hyperparams[1]
            hyperparams_dict['W_0'] = hyperparams[2:8].reshape(2, 3)
    elif inference_model == 'full_metric_gd' and depth == 'two-layer':
        hyperparams_dict['log_alpha_W1'] = hyperparams[0]
        hyperparams_dict['log_alpha_W2_const'] = hyperparams[1]
        hyperparams_dict['log_alpha_W2_stim'] = hyperparams[2]
        hyperparams_dict['log_beta'] = hyperparams[3]
        hyperparams_dict['W1_diag_0'] = hyperparams[4:7]
        hyperparams_dict['W2_0'] = hyperparams[7:13].reshape(2, 3)
    elif inference_model == 'pathway_metric_gd' and depth == 'two-layer':
        hyperparams_dict['log_alpha_const'] = hyperparams[0]
        hyperparams_dict['log_alpha_stim'] = hyperparams[1]
        hyperparams_dict['log_beta'] = hyperparams[2]
        hyperparams_dict['W1_diag_0'] = hyperparams[3:6]
        hyperparams_dict['W2_0'] = hyperparams[6:12].reshape(2, 3)
    elif (inference_model == 'full_metric_gd' or inference_model == 'pathway_metric_gd') and depth == 'one-layer':
        hyperparams_dict['log_alpha_const'] = hyperparams[0]
        hyperparams_dict['log_alpha_stim'] = hyperparams[1]
        hyperparams_dict['log_beta'] = hyperparams[2]
        hyperparams_dict['W_0'] = hyperparams[3:9].reshape(2, 3)
    else:
        # fixed metric GD
        assert inference_model == 'fixed_metric_gd' and depth == 'two-layer', "Layer 1-fixed metric GD is only for 2-layer networks"
        hyperparams_dict['log_alpha_const'] = hyperparams[0]
        hyperparams_dict['log_alpha_stim'] = hyperparams[1]
        hyperparams_dict['log_beta'] = hyperparams[2]
        hyperparams_dict['W1_diag_0'] = hyperparams[3:6]
        hyperparams_dict['W2_0'] = hyperparams[6:12].reshape(2, 3)
    
    return hyperparams_dict
    

def convert_to_array(hyperparams_dict, inference_model):
    assert inference_model in ['gd', 'full_metric_gd', 'pathway_metric_gd', 'fixed_metric_gd']
    
    depth = 'two-layer'
    if 'W_0' in hyperparams_dict:
        depth = 'one-layer'
    
    hyperparams_list = []

    if inference_model == 'gd':
        if depth == 'two-layer':
            hyperparams_list.append(hyperparams_dict['log_alpha'])
            hyperparams_list.append(hyperparams_dict['log_beta'])
            hyperparams_list.extend(hyperparams_dict['W1_diag_0'])
            hyperparams_list.extend(hyperparams_dict['W2_0'].flatten())
        else:
            hyperparams_list.append(hyperparams_dict['log_alpha'])
            hyperparams_list.append(hyperparams_dict['log_beta'])
            hyperparams_list.extend(hyperparams_dict['W_0'].flatten())
    elif inference_model == 'full_metric_gd' and depth == 'two-layer':
        hyperparams_list.append(hyperparams_dict['log_alpha_W1'])
        hyperparams_list.append(hyperparams_dict['log_alpha_W2_const'])
        hyperparams_list.append(hyperparams_dict['log_alpha_W2_stim'])
        hyperparams_list.append(hyperparams_dict['log_beta'])
        hyperparams_list.extend(hyperparams_dict['W1_diag_0'])
        hyperparams_list.extend(hyperparams_dict['W2_0'].flatten())
    elif inference_model == 'pathway_metric_gd' and depth == 'two-layer':
        hyperparams_list.append(hyperparams_dict['log_alpha_const'])
        hyperparams_list.append(hyperparams_dict['log_alpha_stim'])
        hyperparams_list.append(hyperparams_dict['log_beta'])
        hyperparams_list.extend(hyperparams_dict['W1_diag_0'])
        hyperparams_list.extend(hyperparams_dict['W2_0'].flatten())
    elif (inference_model == 'full_metric_gd' or inference_model == 'pathway_metric_gd') and depth == 'one-layer':
        hyperparams_list.append(hyperparams_dict['log_alpha_const'])
        hyperparams_list.append(hyperparams_dict['log_alpha_stim'])
        hyperparams_list.append(hyperparams_dict['log_beta'])
        hyperparams_list.extend(hyperparams_dict['W_0'].flatten())
    else:
        # fixed metric GD
        assert inference_model == 'fixed_metric_gd' and depth == 'two-layer', "Layer 1-fixed metric GD is only for 2-layer networks"
        hyperparams_list.append(hyperparams_dict['log_alpha_const'])
        hyperparams_list.append(hyperparams_dict['log_alpha_stim'])
        hyperparams_list.append(hyperparams_dict['log_beta'])
        hyperparams_list.extend(hyperparams_dict['W1_diag_0'])
        hyperparams_list.extend(hyperparams_dict['W2_0'].flatten())

    hyperparams_arr = jnp.array(hyperparams_list)
    
    return hyperparams_arr


def extract_weights_from_hyperparams(hyperparams, inference_model, return_dict=True):
    assert inference_model in ['gd', 'full_metric_gd', 'pathway_metric_gd', 'fixed_metric_gd']
    
    depth = get_depth(hyperparams, inference_model)

    w = None
    w_start_idx = get_num_alphas(inference_model, depth) + 1

    if depth == 'two-layer':
        w = hyperparams[w_start_idx : w_start_idx + NUM_TWO_LAYER_WEIGHTS]
    else:
        w = hyperparams[w_start_idx : w_start_idx + NUM_ONE_LAYER_WEIGHTS]

    if return_dict:
        if depth == 'two-layer':
            W1_diag_0 = hyperparams[w_start_idx : w_start_idx + W1_NUM_WEIGHTS]
            W2_0 = hyperparams[w_start_idx + W1_NUM_WEIGHTS : w_start_idx + W1_NUM_WEIGHTS + W2_NUM_WEIGHTS].reshape(2, 3)
            w = {'W1_diag' : W1_diag_0, 'W2' : W2_0}
        else:
            W_0 = hyperparams[w_start_idx : w_start_idx + NUM_ONE_LAYER_WEIGHTS].reshape(2, 3)
            w = {'W' : W_0}

    return w


def convert_weights_to_dict(w):
    if len(w) == NUM_ONE_LAYER_WEIGHTS:
        w_dict = {'W' : w.reshape(2, 3)}
    else:
        W1_diag = w[:3]
        W2 = w[3:9].reshape(2, 3)
        w_dict = {'W1_diag' : W1_diag, 'W2' : W2}
    
    return w_dict


def get_depth(hyperparams, inference_model):
    assert inference_model in ['gd', 'full_metric_gd', 'pathway_metric_gd', 'fixed_metric_gd']
    depth = 'two-layer'
    if len(hyperparams) <= ONE_LAYER_MAX_PARAMS:
        depth = 'one-layer'
    
    return depth


def get_num_alphas(inference_model, depth):
    assert inference_model in ['gd', 'full_metric_gd', 'pathway_metric_gd', 'fixed_metric_gd']
    assert depth in ['two-layer', 'one-layer']
    num_alphas = None 

    if depth == 'two-layer':
        if inference_model == 'gd':
            num_alphas = 1
        elif inference_model == 'full_metric_gd':
            num_alphas = 3
        elif inference_model == 'pathway_metric_gd' or inference_model == 'fixed_metric_gd':
            num_alphas = 2
    else:
        assert inference_model != 'fixed_metric_gd', "Layer 1-fixed metric GD is only for 2-layer networks"
        if inference_model == 'gd':
            num_alphas = 1
        elif inference_model == 'full_metric_gd' or inference_model == 'pathway_metric_gd':
            num_alphas = 2
    
    return num_alphas


def generate_init_weights(depth, w0_init_scale, key, w0_init_guess=None, noise_level=0.01, return_array=False,
                          layer_two_multiplier=1.):
    assert depth in ['two-layer', 'one-layer']
    w0 = None 
    subkeys = jr.split(key, 5) 

    if depth == 'two-layer':
        W1_diag_0 = w0_init_scale * jr.normal(subkeys[1], (3,)) 
        W2_0 = w0_init_scale * layer_two_multiplier * jr.normal(subkeys[2], (2, 3)) 
        if w0_init_guess is not None:
            W1_diag_0 = w0_init_guess['W1_diag'] * (1 + noise_level * jr.normal(subkeys[3], (3,)))
            W2_0 = w0_init_guess['W2'] * (1 + noise_level * jr.normal(subkeys[4], (2, 3)))
        
        if return_array:
            w0 = jnp.concatenate([W1_diag_0, W2_0.flatten()])
        else:
            w0 = {'W1_diag' : W1_diag_0, 'W2' : W2_0}

    else:
        W_0 = w0_init_scale * jr.normal(subkeys[1], (2, 3)) 
        if w0_init_guess is not None:
            W_0 = w0_init_guess['W'] * (1 + noise_level * jr.normal(subkeys[2], (2, 3)))
        
        if return_array:
            w0 = W_0.flatten()
        else:
            w0 = {'W' : W_0}
    
    return w0



def generate_init_hyperparams(inference_model, depth, alpha_init_scale, beta_init_scale, w0_init_scale, key,
                              alpha_init_guess=None, beta_init_guess=None, w0_init_guess=None, noise_level=0.01):
    subkeys = jr.split(key, 6)
    num_alphas = get_num_alphas(inference_model, depth)

    beta0 = beta_init_scale * jnp.abs(jr.normal(subkeys[1], (1,)))
    if beta_init_guess is not None:
        assert beta_init_guess > 0.0
        beta0 = beta_init_guess * (1 + noise_level * jr.normal(subkeys[2], (1,)))

    alpha_inits = alpha_init_scale * jnp.abs(jr.normal(subkeys[3], (num_alphas,))) 
    if alpha_init_guess is not None:
        assert all(alpha_init_guess > 0.0)
        alpha_inits = alpha_init_guess * (1 + noise_level * jr.normal(subkeys[4], (num_alphas,))) 

    log_beta0 = jnp.log(beta0)
    log_alpha_inits = jnp.log(alpha_inits)

    w0 = generate_init_weights(depth, w0_init_scale, subkeys[5], w0_init_guess=w0_init_guess, noise_level=noise_level, return_array=True)

    init_hyperparams_arr = jnp.concatenate([log_alpha_inits, log_beta0, w0])
    return init_hyperparams_arr


def unpack_hyperparams(hyperparams, inference_model, fixed_alpha_W1=None):
    # assumes hyperparams is a list
    assert inference_model in ['gd', 'full_metric_gd', 'pathway_metric_gd', 'fixed_metric_gd']

    depth = get_depth(hyperparams, inference_model)

    if inference_model != 'fixed_metric_gd':
        assert fixed_alpha_W1 is None

    log_beta = None
    log_lrs = None
    w0 = None 

    if depth == 'two-layer':
        W1_diag_0, W2_0 = None, None 
        log_alphas_W1, log_alphas_W2 = None, None 
        if inference_model == 'gd':
            log_alpha = hyperparams[0]
            log_beta = hyperparams[1]
            W1_diag_0 = hyperparams[2:5]
            W2_0 = hyperparams[5:11].reshape(2, 3)

            log_alphas_W1 = log_alpha
            log_alphas_W2 = log_alpha
        
        elif inference_model == 'full_metric_gd':
            log_alpha_W1 = hyperparams[0]
            log_alpha_W2_const = hyperparams[1]
            log_alpha_W2_stim = hyperparams[2]
            log_beta = hyperparams[3]
            W1_diag_0 = hyperparams[4:7]
            W2_0 = hyperparams[7:13].reshape(2, 3)

            # Use same learning rate (alpha_W2_stim) for both stimulus-related weights
            log_alphas_W2 = jnp.array([log_alpha_W2_const, log_alpha_W2_stim, log_alpha_W2_stim]).reshape(1, -1)
            log_alphas_W1 = log_alpha_W1
        
        elif inference_model == 'pathway_metric_gd' or inference_model == 'fixed_metric_gd':
            log_alpha_const = hyperparams[0]
            log_alpha_stim = hyperparams[1]
            log_beta = hyperparams[2]
            W1_diag_0 = hyperparams[3:6]
            W2_0 = hyperparams[6:12].reshape(2, 3)

            if inference_model == 'pathway_metric_gd':
                log_alphas_W1 = jnp.array([log_alpha_const, log_alpha_stim, log_alpha_stim])
            else:
                log_alphas_W1 = jnp.log(fixed_alpha_W1)

            log_alphas_W2 = jnp.array([log_alpha_const, log_alpha_stim, log_alpha_stim]).reshape(1, -1)

        log_lrs = {'W1_diag' : log_alphas_W1, 'W2' : log_alphas_W2}
        w0 = {'W1_diag' : W1_diag_0, 'W2' : W2_0}
    else:
        W_0 = None 
        alphas = None
        assert inference_model != 'fixed_metric_gd', "Layer 1-fixed metric GD is only for 2-layer networks"

        if inference_model == 'gd':
            log_alpha = hyperparams[0]
            log_beta = hyperparams[1]
            W_0 = hyperparams[2:8].reshape(2, 3)

            alphas = log_alpha

        elif inference_model == 'full_metric_gd' or inference_model == 'pathway_metric_gd':
            log_alpha_const = hyperparams[0]
            log_alpha_stim = hyperparams[1]
            log_beta = hyperparams[2]
            W_0 = hyperparams[3:9].reshape(2, 3)

            # Use same learning rate (alpha_stim) for both stimulus-related weights
            alphas = jnp.array([log_alpha_const, log_alpha_stim, log_alpha_stim]).reshape(1, -1)
        
        log_lrs = {'W' : alphas}
        w0 = {'W' : W_0}
    
    return log_lrs, log_beta, w0


@partial(jit, static_argnames=('nln',))
def model_forward(w, x, nln):
    if "W" in w:
        # one-layer network, no nonlinearity
        return w["W"] @ x

    elif "W1_diag" in w:
        # two-layer network, possibly with nonlinearity
        h = w["W1_diag"] * x
        h = nln(h)
        return w["W2"] @ h

    else:
        raise KeyError("Unrecognized parameter structure.")

batched_model_forward = vmap(model_forward, in_axes=(0, 0, None))


@jit
def log_likelihood(out, choice_idx, log_beta):
    '''
    out        : the current output (Q-values) of the model
    choice_idx : the choice made by the agent (0 = R, 1 = L)
    log_beta   : log of the inverse-temperature (softmax gain) parameter
    '''
    Q_y = out[choice_idx]
    Q_opp = out[1-choice_idx]
    p_y_given_w_and_x = 1.0 / (1.0 + jnp.exp(-jnp.exp(log_beta) * (Q_y - Q_opp)))
    ll = jnp.log(p_y_given_w_and_x)
    return ll

# Loss: squared error loss
@partial(jit, static_argnames=('nln',))
def squared_error_loss_fn(w, x, y_choice, rew, nln):
    out = model_forward(w, x, nln)
    Q_y = out[y_choice]
    return 0.5 * (Q_y - rew)**2

@jit 
def apply_learning_updates(w, grads, log_lrs):
    return jax.tree_util.tree_map(lambda p, g, log_lr: p - jnp.exp(log_lr) * g, w, grads, log_lrs)




# ── Training function ─────────────────────────────────────────────────────────

def train_gd_network(nln_name, params, inputs, targets, lr, beta,
                     key, num_grading_levels=4):
    """
    Train a minimal neural network on a simple binary decision-making task.

    nln_name            : string specifying the nonlinearity (two-layer only)
    params              : model parameters dict, e.g. {'W1_diag': ..., 'W2': ...}
    inputs              : shape (num_trials, 3)  — (const, R, L)
    targets             : shape (num_trials, 1)  — 0=R, 1=L
    lr                  : learning rate
    beta                : inverse temperature for softmax policy
    key                 : JAX RNG key
    num_grading_levels  : number of contrast levels per side (excluding zero)
    """
    num_trials = inputs.shape[0]
    nln = get_nonlinearity(nln_name)
    full_input_set = generate_full_input_set(num_grading_levels)  # (2L+1, 3)

    # ── Inner helpers (no @jit: they close over Python values and are
    #    recreated each call, so JIT would retrace every time) ────────────────

    @jit
    def softmax_policy(out, key):
        p_R = jax.nn.sigmoid(beta * (out[0] - out[1]))
        choice_idx = jnp.where(jr.bernoulli(key, p=1.0 - p_R), 1, 0)
        return p_R, choice_idx

    @jit
    def get_reward(params, x, target, key):
        key_policy, key_rand = jr.split(key)
        out = model_forward(params, x, nln)
        _, choice_idx = softmax_policy(out, key_policy)

        is_not_zero = jnp.minimum(2 * num_grading_levels * jnp.sum(x[1:]), 1.0)
        correct = jnp.where(choice_idx == target, 1, 0)
        rand_rew = jr.bernoulli(key_rand, 0.5).astype(jnp.int32)
        rew = correct * is_not_zero + rand_rew * (1 - is_not_zero)
        return jax.lax.stop_gradient(rew), choice_idx

    @jit
    def gd_step(params, x, y_choice, rew):
        loss, grads = jax.value_and_grad(squared_error_loss_fn)(
            params, x, y_choice, rew, nln
        )
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        return params, loss

    @jit
    def zero_contrast_bias(params):
        out = model_forward(params, jnp.array([1.0, 0.0, 0.0]), nln)
        return jax.nn.sigmoid(beta * (out[0] - out[1])) - 0.5

    @jit
    def psychometric_data(params):
        # vmap over all contrast levels at once instead of a Python loop
        outs = vmap(lambda x: model_forward(params, x, nln))(full_input_set)
        return jax.nn.sigmoid(beta * (outs[:, 0] - outs[:, 1]))

    # ── Training loop ─────────────────────────────────────────────────────────

    losses, rewards, outputs, q_vals, zero_contrast_biases = [], [], [], [], []
    params_list = [params]
    psychometrics = np.zeros((num_trials, 2 * num_grading_levels + 1))

    for n in range(num_trials):
        key, subkey = jr.split(key)
        x, target = inputs[n], targets[n, 0]

        rew, choice_idx = get_reward(params, x, target, subkey)
        rewards.append(int(rew))
        outputs.append(choice_idx)
        q_vals.append(model_forward(params, x, nln))

        params, loss = gd_step(params, x, choice_idx, rew)
        params_list.append(params)

        zero_contrast_biases.append(zero_contrast_bias(params))
        psychometrics[n] = psychometric_data(params)
        losses.append(float(loss))

        if (n + 1) % 250 == 0:
            print(f'step {n+1}, loss: {loss:.6f}')

    r_slopes = psychometrics[:, -1] - psychometrics[:, num_grading_levels]
    l_slopes = psychometrics[:, num_grading_levels] - psychometrics[:, 0]

    Rs = 1. - jnp.asarray(outputs) # convert from 0=R, 1=L to 1=R, 0=L for easier interpretation
    ma_bias = 2 * jnp.mean(Rs) - 1
    print(f'MA bias (over all trials): {ma_bias:.4f}')

    return params, losses, {
        'params':        params,
        'losses':        losses,
        'rewards':       rewards,
        'outputs':       outputs,
        'q_vals':        q_vals,
        'biases':        zero_contrast_biases,
        'psychometrics': psychometrics,
        'slope_diffs':   r_slopes - l_slopes,
        'params_list':   params_list,
        'ma-bias':       ma_bias
    }


def compute_channel_contribution(w1s, W2s, path_idx, nln, right_or_left):
    '''
    Arguments

    w1s        : layer 1 diagonal weights across learning (shape: T x 3)
    W2s        : layer 2 weights across learning (shape: T x 2)
    path_idx   : index of the pathway (0 - constant, 1 - right, 2 - left)
    nln        : nonlinearity used
    '''
    assert right_or_left in ['right', 'left']
    T = w1s.shape[0]

    w1_path = w1s[:, path_idx].reshape(T, 1)      # (T, 1)
    W2_path = W2s[:, :, path_idx]                 # (T, 2)

    input = None
    if right_or_left == 'right':
        input = FULL_CONTRAST_RIGHT_INPUT
    else:
        input = FULL_CONTRAST_LEFT_INPUT

    input_unit = input[path_idx]

    out = W2_path * nln(w1_path * input_unit)    # (T, 2)

    return out


def gather_input_contributions(ws, nln):
    T = ws.shape[0]
    network_stats = {}

    if ws.shape[1] == NUM_ONE_LAYER_WEIGHTS:
        W = ws.reshape(T, 2, 3)
        network_stats['right_contribs'] = W * FULL_CONTRAST_RIGHT_INPUT.reshape(1, 1, -1)
        network_stats['left_contribs'] = W * FULL_CONTRAST_LEFT_INPUT.reshape(1, 1, -1)
        network_stats['right_diffs'] = network_stats['right_contribs'][:, 0, :] - network_stats['right_contribs'][:, 1, :]
        network_stats['left_diffs'] = network_stats['left_contribs'][:, 0, :] - network_stats['left_contribs'][:, 1, :]
    
    else:
        w1s = ws[:, :3]
        W2s = ws[:, 3:].reshape(T, 2, 3)

        for ch in ['right', 'left']:
            const_out = compute_channel_contribution(w1s, W2s, 0, nln, ch)
            right_out = compute_channel_contribution(w1s, W2s, 1, nln, ch)
            left_out = compute_channel_contribution(w1s, W2s, 2, nln, ch)
            input_contribs = jnp.stack([const_out, right_out, left_out], axis=-1)   # (T, 2, 3)
            loading_diffs = input_contribs[:, 0, :] - input_contribs[:, 1, :]
            network_stats[f'{ch}_contribs'] = input_contribs
            network_stats[f'{ch}_diffs'] = loading_diffs
    
    return network_stats


def visualize_channel_contributions(ws, nln, filepath=None, ws_true=None):
    '''
    Arguments

    ws  : weights across learning (shape: T x 9)
    nln : nonlinearity used
    '''
    network_stats = gather_input_contributions(ws, nln)
    right_input_contribs = network_stats['right_contribs']
    right_loading_diffs = network_stats['right_diffs']
    left_input_contribs = network_stats['left_contribs']
    left_loading_diffs = network_stats['left_diffs']

    true_right_input_contribs, true_right_loading_diffs, true_left_input_contribs, true_left_loading_diffs = None, None, None, None
    if ws_true is not None:
        true_network_stats = gather_input_contributions(ws_true, nln)
        true_right_input_contribs = true_network_stats['right_contribs']
        true_right_loading_diffs = true_network_stats['right_diffs']
        true_left_input_contribs = true_network_stats['left_contribs']
        true_left_loading_diffs = true_network_stats['left_diffs']

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 8))
    fig.suptitle('Right Input Contributions')
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(right_input_contribs[:, i, j], color='green', lw=3., linestyle='dashed')
            if ws_true is not None:
                ax[i, j].plot(true_right_input_contribs[:, i, j], color='green', lw=2.2)
            ax[i, j].axhline(0, lw=2., color='blue', linestyle=':')
    
    ax[0, 0].set_ylabel('Q_R')
    ax[1, 0].set_ylabel('Q_L')
    
    if filepath is None:
        plt.show()
    else:
        plt.savefig(f'{filepath}/right_input_contributions.png')
        # plt.close()

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 8))
    fig.suptitle('Left Input Contributions')
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(left_input_contribs[:, i, j], color='green', lw=3., linestyle='dashed')
            if ws_true is not None:
                ax[i, j].plot(true_left_input_contribs[:, i, j], color='green', lw=2.2)
            ax[i, j].axhline(0, lw=2., color='blue', linestyle=':')
    
    ax[0, 0].set_ylabel('Q_R')
    ax[1, 0].set_ylabel('Q_L')
    
    if filepath is None:
        plt.show()
    else:
        plt.savefig(f'{filepath}/left_input_contributions.png')
        # plt.close()


    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 8))
    fig.suptitle('Input Loading Differences')
    for i in range(3):
        ax[0, i].plot(right_loading_diffs[:, i], color='green', lw=3., linestyle='dashed')
        ax[0, i].axhline(0, lw=2., color='blue', linestyle=':')

        ax[1, i].plot(left_loading_diffs[:, i], color='darkviolet', lw=3., linestyle='dashed')
        ax[1, i].axhline(0, lw=2., color='blue', linestyle=':')

        if ws_true is not None:
            ax[0, i].plot(true_right_loading_diffs[:, i], color='green', lw=2.2)
            ax[1, i].plot(true_left_loading_diffs[:, i], color='darkviolet', lw=2.2)
    
    ax[0, 0].set_ylabel('$Q_R - Q_L$ \\ Right Input')
    ax[1, 0].set_ylabel('$Q_R - Q_L$ \\ Left Input')
    
    if filepath is None:
        plt.show()
    else:
        plt.savefig(f'{filepath}/input_loading_differences.png')
        # plt.close()
    
    return


def visualize_nln_usage(w, nln, num_gradings=10, title=None, filepath=None):
    '''
    Arguments
    
    w      :  PyTree of NN weights
    nln    :  nonlinearity used 
    inputs :  inputs tested on (shape: num_inputs x 3)
    '''
    delta = 1./num_gradings
    input_units = jnp.arange(0, 1+delta, delta)
    W1_diag = w['W1_diag']

    const_channel_h = nln(W1_diag[0])
    right_hs = nln(W1_diag[1] * input_units)
    left_hs = nln(W1_diag[2] * input_units)

    # plt.figure()
    # plt.scatter([1.], [const_channel_h], color='orange', s=100, label='Constant unit')
    # plt.scatter(input_units, right_hs, color='forestgreen', s=100, label='Right units')
    # plt.scatter(input_units, left_hs, color='darkviolet', s=100, label='Left units')
    # plt.legend()

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(input_units, right_hs, color='#2CA02C', s=30, label='Right units', zorder=3)
    ax.scatter(input_units, left_hs,  color='#9467BD', s=30, label='Left units',  zorder=3)
    ax.scatter([1.], [const_channel_h], color='#F4A736', s=50, marker='*',
               label='Constant unit', zorder=4)

    # Plot the raw nonlinearity for reference
    x_ref = jnp.linspace(input_units.min(), input_units.max(), 200)
    ax.plot(x_ref, nln(x_ref), color='gray', lw=0.9, linestyle='--',
            alpha=0.6, zorder=1, label='$\sigma(x)$')

    ax.set_xlabel('Input', fontsize=13)
    ax.set_ylabel('Hidden Unit Activation', fontsize=13)

    if title is None:
        title = 'Nonlinearity Usage'
    ax.set_title(title, fontsize=14)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=11, length=3, width=0.8)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False)

    plt.tight_layout()
    if filepath is None:
        plt.show()
    else:
        fig.savefig(f'{filepath}', dpi=300, bbox_inches='tight')


# require 1d array of alphas, 1d array of betas, and 1 array of nlls
def visualize_alpha_beta_nlls(alphas, betas, nlls, vmin, vmax,
                              alpha_upper_lim=None, beta_upper_lim=None):
    alphas = np.asarray(alphas).reshape(-1)
    betas = np.asarray(betas).reshape(-1)
    nlls = np.asarray(nlls).reshape(-1)
    assert len(alphas) == len(betas)
    assert len(betas) == len(nlls)

    plt.figure(figsize=(6, 5))

    sc = plt.scatter(
        alphas,
        betas,
        c=nlls,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=50
    )

    if alpha_upper_lim is not None:
        plt.xlim(right=alpha_upper_lim)
    
    if beta_upper_lim is not None:
        plt.ylim(right=beta_upper_lim)

    plt.colorbar(sc, label="NLL")
    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.title("Results of Various Optimization Seeds")
    plt.tight_layout()
    plt.show()
    return
    



# inference_model = 'gd'
# depth = 'one-layer'
# alpha_init_scale = 1e-1
# beta_init_scale = 1e1
# w0_init_scale = 1e-1
# nln = jax.nn.identity
# fixed_alpha_W1 = None
# key = jr.PRNGKey(32)

# num_alphas = get_num_alphas(inference_model, depth)
# print(f'num alphas: {num_alphas}')
# init_hyperparams_arr = generate_init_hyperparams(inference_model, depth, alpha_init_scale, beta_init_scale, w0_init_scale, key,
#                                                  alpha_init_guess=None, beta_init_guess=None, w0_init_guess=None, noise_level=0.01)


# log_lrs, log_beta, w0 = unpack_hyperparams(init_hyperparams_arr, inference_model, fixed_alpha_W1=fixed_alpha_W1)
# print(log_lrs)
# print(log_beta)
# print(w0)

# visualize_nln_usage(w0, nln, num_gradings=10)

