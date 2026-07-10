# --- path shim: allow running this legacy script standalone after the move to legacy/ ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path shim ---
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax.tree_util import tree_map
import optax
import optax.tree_utils as otu
from functools import partial
import numpy as np

from inference_utils import *
from globals import ERF, DEPTH_TO_NUM_WEIGHTS

@jit
def run_static_model(w, inputs, choices):
    """
    Arguments:
    - w: weights to infer
    - inputs
    - choices
    - rewards

    Returns:
    - total negative log-likelihood of the data given the weights w
    """

    T = len(inputs)

    outs = inputs @ w 
    P_rights = jax.nn.sigmoid(outs)
    P_lefts = 1. - P_rights
    probs = jnp.column_stack((P_rights, P_lefts))
    data_probs = probs[jnp.arange(T), choices]

    total_nll = -jnp.sum(jnp.log(data_probs))

    return total_nll

batched_run_static_model = jit(vmap(run_static_model, in_axes=(None, 0, 0)))


def compute_static_model_nll(w, inputs, choices, batched):
    nll = None 
    if batched:
        nlls = batched_run_static_model(w, inputs, choices)
        nll = jnp.mean(nlls)
    else:
        nll = run_static_model(w, inputs, choices)

    return nll

def optax_fit_params(w, inputs, choices, batched=False, max_iters=500, tol=1e-2):

    meta_loss_fn = partial(compute_static_model_nll, inputs=inputs, choices=choices, batched=batched)
    opt = optax.lbfgs(memory_size=50) 
    value_and_grad_fun = optax.value_and_grad_from_state(meta_loss_fn)

    # Jit the step function for faster optimization iterations
    @jit
    def step(carry):
        hyperparams, state = carry
        value, grad = value_and_grad_fun(hyperparams, state=state)
        updates, state = opt.update(
            grad, state, hyperparams, value=value, grad=grad, value_fn=meta_loss_fn
        )
        hyperparams = optax.apply_updates(hyperparams, updates)
        return hyperparams, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iters) & (err >= tol))

    init_carry = (w, opt.init(w))
    final_hyperparams, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_hyperparams, final_state

def generate_init_w(init_scale, key, w_init_guess=None, noise_level=0.01):
    w = init_scale * jr.normal(key, (3,)) 
    if w_init_guess is not None:
        w = w_init_guess * (1 + noise_level * jr.normal(key, (3,)))
    return w


if __name__ == '__main__':
    results_dict = pkl.load(open(f'witten_data_GD_analysis/fitted_models/gd_linear_one-layer/fip14_results_one-layer_gd_linear.pkl', 'rb'))
    inputs = results_dict['inputs']
    outputs = results_dict['outputs'].astype(jnp.int32)
    rewards = results_dict['rewards']
    x0 = results_dict['inferred_hyperparams'].astype(jnp.float64)

    key = jr.PRNGKey(18)
    max_iters = 5_000
    w = generate_init_w(1e-1, key)
    w_inferred, final_state = optax_fit_params(w, inputs, outputs, max_iters=max_iters, tol=1e-2)
    iter_count = int(otu.tree_get(final_state, 'count'))

    print(f'Iters: {iter_count}/{max_iters}')
    print(w_inferred)

    nll = run_static_model(w_inferred, inputs, outputs)
    print(f'Inferred NLL: {nll}')


