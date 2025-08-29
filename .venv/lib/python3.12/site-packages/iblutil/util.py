import uuid
from itertools import takewhile
from os import scandir
from pathlib import Path
import collections
import colorlog
import copy
import logging
import sys
from typing import Union, Iterable, Sequence

import numpy as np

log = logging.getLogger('__name__')

LOG_FORMAT_STR = '%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4d %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_COLORS = {'DEBUG': 'green', 'INFO': 'cyan', 'WARNING': 'bold_yellow', 'ERROR': 'bold_red', 'CRITICAL': 'bold_purple'}
SIZE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']


def Listable(t):
    """Return a typing.Union if the input and sequence of input."""
    return Union[t, Sequence[t]]


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self, deep=False):
        """Return a new Bunch instance which is a copy of the current Bunch instance.

        Parameters
        ----------
        deep : bool
            If True perform a deep copy (see notes). By default a shallow copy is returned.

        Returns
        -------
        Bunch
            A new copy of the Bunch.

        Notes
        -----
        - A shallow copy constructs a new Bunch object and then (to the extent possible) inserts
        references into it to the objects found in the original.
        - A deep copy constructs a new Bunch and then, recursively, inserts copies into it of the
         objects found in the original.
        """
        return copy.deepcopy(self) if deep else Bunch(super(Bunch, self).copy())

    def save(self, npz_file, compress=False):
        """
        Saves a npz file containing the arrays of the bunch.

        :param npz_file: output file
        :param compress: bool (False) use compression
        :return: None
        """
        if compress:
            np.savez_compressed(npz_file, **self)
        else:
            np.savez(npz_file, **self)

    @staticmethod
    def load(npz_file):
        """
        Loads a npz file containing the arrays of the bunch.

        :param npz_file: output file
        :return: Bunch
        """
        if not Path(npz_file).exists():
            raise FileNotFoundError(f'{npz_file}')
        return Bunch(np.load(npz_file))


def _iflatten(x):
    result = []
    for el in x:
        if isinstance(el, collections.abc.Iterable) and not (isinstance(el, str) or isinstance(el, dict)):
            result.extend(_iflatten(el))
        else:
            result.append(el)
    return result


def _gflatten(x):
    def iselement(e):
        return not (isinstance(e, collections.abc.Iterable) and not (isinstance(el, str) or isinstance(el, dict)))

    for el in x:
        if iselement(el):
            yield el
        else:
            yield from _gflatten(el)


def flatten(x, generator=False):
    """
    Flatten a nested Iterable excluding strings and dicts.

    Converts nested Iterable into flat list. Will not iterate through strings or
    dicts.

    :return: Flattened list or generator object.
    :rtype: list or generator
    """
    return _gflatten(x) if generator else _iflatten(x)


def range_str(values: iter) -> str:
    """
    Given a list of integers, returns a terse string expressing the unique values.

    Example:
        indices = [0, 1, 2, 3, 4, 7, 8, 11, 15, 20]
        range_str(indices)
        >> '0-4, 7-8, 11, 15 & 20'
    :param values: An iterable of ints
    :return: A string of unique value ranges
    """
    trial_str = ''
    values = list(set(values))
    for i in range(len(values)):
        if i == 0:
            trial_str += str(values[i])
        elif values[i] - (values[i - 1]) == 1:
            if i == len(values) - 1 or values[i + 1] - values[i] > 1:
                trial_str += f'-{values[i]}'
        else:
            trial_str += f', {values[i]}'
    # Replace final comma with an ampersand
    k = trial_str.rfind(',')
    if k > -1:
        trial_str = f'{trial_str[:k]} &{trial_str[k + 1 :]}'
    return trial_str


def setup_logger(name='ibl', level=logging.NOTSET, file=None, no_color=False):
    """Set up a log for IBL packages.

    Uses date time, calling function and distinct colours for levels.
    Sets the name if not set already and add a stream handler.
    If the stream handler already exists, does not duplicate.
    The naming/level allows not to interfere with third-party libraries when setting level.

    Parameters
    ----------
    name : str
        Log name, should be set to the root package name for consistent logging throughout the app.
    level : str, int
        The logging level (defaults to NOTSET, which inherits the parent log level)
    file : bool, str, pathlib.Path
        If True, a file handler is added with the default file location, otherwise a log file path
        may be passed.
    no_color : bool
        If true the colour log is deactivated.  May be useful when directing the std out to a file.

    Returns
    -------
    logging.Logger, logging.RootLogger
        The configured log.
    """
    log = logging.getLogger() if not name else logging.getLogger(name)
    log.setLevel(level)
    fkwargs = {'no_color': True} if no_color else {'log_colors': LOG_COLORS}
    # check existence of stream handlers before adding another
    if not any(map(lambda x: x.name == f'{name}_auto', log.handlers)):
        # need to remove any previous default Stream handler configured on stderr
        # to not duplicate output
        for h in log.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream.name == '<stderr>' and h.level == 0 and h.name is None:
                log.removeHandler(h)
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s' + LOG_FORMAT_STR, LOG_DATE_FORMAT, **fkwargs))
        stream_handler.name = f'{name}_auto'
        log.addHandler(stream_handler)
    # add the file handler if requested, but check for duplicates
    if not any(map(lambda x: x.name == f'{name}_file', log.handlers)):
        if file is True:
            log_to_file(log=name)
        elif file:
            log_to_file(filename=file, log=name)
    return log


def log_to_file(log='ibl', filename=None):
    """
    Save log information to a given filename in '.ibl_logs' folder (in home directory).

    Parameters
    ----------
    log : str, logging.Logger
        The log (name or object) to add file handler to.
    filename : str, Pathlib.Path
        The name of the log file to save to.

    Returns
    -------
    logging.Logger
        The log with the file handler attached.
    """
    if isinstance(log, str):
        log = logging.getLogger(log)
    if filename is None:
        filename = Path.home().joinpath('.ibl_logs', log.name)
    elif not Path(filename).is_absolute():
        filename = Path.home().joinpath('.ibl_logs', filename)
    filename.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(filename, encoding='utf-8')
    file_format = logging.Formatter(LOG_FORMAT_STR, LOG_DATE_FORMAT)
    file_handler.setFormatter(file_format)
    file_handler.name = f'{log.name}_file'
    log.addHandler(file_handler)
    log.info(f'File log initiated {file_handler.name}')
    return log


def rrmdir(folder: Path, levels: int = 0):
    """
    Recursively remove a folder and its parents up to a defined level - if they are empty.

    Parameters
    ----------
    folder : pathlib.Path
        The path to a folder at which to start the recursion.
    levels : int
        Recursion level, i.e. the number of parents to delete, relative to `folder`.
        Defaults to 0 - which has the same effect as `pathlib.Path.rmdir` except that it won't
        raise an OSError if the directory is not empty.

    Returns
    -------
    list of pathlib.Path
        A list of folders that were recursively removed.

    Raises
    ------
    FileNotFoundError
        If `folder` does not exist.
    PermissionError
        Insufficient privileges or folder in use by another process.
    NotADirectoryError
        The folder provided is most likely a file.
    """
    try:  # a sorted list of absolute nested folder paths
        to_remove = (folder, *folder.parents[:levels])  # py >= 3.9
    except TypeError:  # py <= 3.8 compatible
        to_remove = (folder, *[folder.parents[n] for n in range(levels)])
    # filter list to those that are empty; if statement always true as rmdir returns None
    return [f for f in takewhile(lambda f: not any(f.iterdir()), to_remove) if not f.rmdir()]


def dir_size(directory: str | Path, follow_symlinks: bool = False) -> int:
    """
    Calculate the total size of a directory including all its subdirectories and files.

    Parameters
    ----------
    directory : str or Path
        The path to the directory for which the size needs to be calculated.
    follow_symlinks : bool, optional
        Whether to follow symbolic links when calculating the size. Default is False.

    Returns
    -------
    int
        The total size of the directory in bytes.
    """
    total_bytes = 0
    with scandir(directory) as it:
        for entry in it:
            if entry.is_symlink() and not follow_symlinks:
                continue
            elif entry.is_dir():
                total_bytes += dir_size(entry.path, follow_symlinks)
            elif entry.is_file():
                total_bytes += entry.stat().st_size
    return total_bytes


def format_bytes(size_bytes: int, decimals: int = 1) -> str:
    """
    Convert a size in bytes into a human-readable string using binary units.

    The function scales the input size using powers of 1024, returning a
    formatted string with the appropriate unit (e.g., B, KB, MB, ...).

    Parameters
    ----------
    size_bytes : int
        The size in bytes to convert.
    decimals : int, optional
        The number of decimal places for units KB and above. Defaults to 1.

    Returns
    -------
    str
        The human-readable string representation of the size.
    """
    scaled_size = size_bytes
    for exponent, unit in enumerate(SIZE_UNITS):
        if scaled_size < 1024:
            break
        scaled_size >>= 10
    else:
        exponent = len(SIZE_UNITS) - 1
        unit = SIZE_UNITS[exponent]
    value = size_bytes / (1024**exponent)
    fmt = f'{{:.{decimals}f}} ' if exponent > 0 else '{:.0f} '
    return fmt.format(value) + unit


def get_mac() -> str:
    """
    Fetch the machine's unique MAC address formatted according to IEEE 802 specifications.

    Returns
    -------
    str
        The MAC address of the device formatted in six groups of two
        hexadecimal digits separated by hyphens in transmission order
        (e.g., 'BA-DB-AD-C0-FF-EE').
    """
    return uuid.getnode().to_bytes(6, 'big').hex('-').upper()


def ensure_list(value, exclude_type=(str, dict)):
    """Ensure input is a list.

    Wraps `value` in a list if not already an iterator or if it is a member of specific
    iterable classes.

    To allow users the option of passing a single value or multiple values, this function
    will wrap the former in a list and by default will consider str and dict instances as
    a single value. This function is useful because it treats tuples, lists, sets, and
    generators all as 'lists', but not dictionaries and strings.

    Parameters
    ----------
    value : any
        Input to ensure list.
    exclude_type : tuple, optional
        A list of iterable classes to wrap in a list.

    Returns
    -------
    Iterable
        Either `value` if iterable and not in `exclude_type` list, or `value` wrapped in a list.
    """
    return [value] if isinstance(value, exclude_type) or not isinstance(value, Iterable) else value
