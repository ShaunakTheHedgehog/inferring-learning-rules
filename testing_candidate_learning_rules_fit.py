import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax import debug
from jax import hessian
from jax.tree_util import tree_map
import optax
import optax.tree_utils as otu

from jaxopt import ScipyMinimize, ScipyBoundedMinimize, BFGS
from functools import partial
import numpy as np

import pickle as pkl

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pdb

# from flax import linen as nn
import functools
import random
import math
from deep_linear_net import cortico_striatal_net, random_init_params, fit_deep_net
from utils import generate_multisession_graded_data, calculate_model_agnostic_bias, calculate_session_accs, flatten_params, turn_pytree_to_arr

