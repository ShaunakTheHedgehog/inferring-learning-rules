import jax

ONE_LAYER_MAX_PARAMS = 10
W1_NUM_WEIGHTS = 3
W2_NUM_WEIGHTS = 6
NUM_ONE_LAYER_WEIGHTS = 6
NUM_TWO_LAYER_WEIGHTS = 9
NUM_BETAS = 1 
SIGMOID = jax.nn.sigmoid 
ERF = jax.lax.erf
ID = jax.nn.identity

DEPTH_TO_NUM_WEIGHTS = {
    'one-layer': NUM_ONE_LAYER_WEIGHTS,
    'two-layer': NUM_TWO_LAYER_WEIGHTS
}

FULL_CONTRAST_RIGHT_INPUT = jax.numpy.array([1., 1., 0.])
FULL_CONTRAST_LEFT_INPUT = jax.numpy.array([1., 0., 1.])

NUM_FINAL_SESSIONS = 3

EXPERT_THRESHOLD = 0.7

BIAS_THRESHOLD = 0.2

NUM_GRADING_LEVELS = 16
WITTEN_DATA_CONTRAST_LEVELS = [0.0625, 0.125, 0.25, 1.0]

RIDGE_LAMBDA = 0.5
RIDGE_ALPHABETA = 0.5

MU_ALPHA = float(jax.numpy.log(1e-3))
MU_BETA = float(jax.numpy.log(1e0))

DATASET_TO_CHOICE_KEYWORD = {'witten': 'outputs', 'saxe': 'choices'}

MODELS = ['1L', '2L', '2NL']

MODEL_CLASS_INFO = {'1-Layer Linear': ['gd', 'one-layer', 'linear', NUM_ONE_LAYER_WEIGHTS+2, 'blue'], 
                    '2-Layer Linear': ['gd', 'two-layer', 'linear', NUM_TWO_LAYER_WEIGHTS+2, 'orange'],
                    '2-Layer Nonlinear': ['gd', 'two-layer', 'erf', NUM_TWO_LAYER_WEIGHTS+2, 'green']}

MODEL_COLORS = ['#2166AC', '#F4A736', '#2CA02C']

MODEL_PLOT_NAMES = ['1-Layer Linear', '2-Layer Linear', '2-Layer Nonlinear']
OLDER_MODEL_NAMES = ['one-layer linear gd', 'two-layer linear gd', 'two-layer erf gd']

SAXE_BAD_IDXS = ['ALK068', 'ALK078', 'DAP017', 'DAP033', 'MFE015']


WITTEN_NUM_MICE = 35

