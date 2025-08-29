import json
from typing import Any, List, Tuple

import pandas as pd


def read(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _write(file, data, mode):
    with open(file, mode) as f:
        for obj in data:
            f.write(json.dumps(obj) + '\n')


def write(file, data):
    _write(file, data, 'w+')


def append(file, data):
    _write(file, data, 'a')


def load_task_jsonable(jsonable_file, offset: int = None) -> Tuple[pd.DataFrame, List[Any]]:
    """
    Reads in a task data jsonable file and returns a trials dataframe and a bpod data list.

    Parameters
    ----------
    jsonable_file : str, pathlib.Path
        Full path to jsonable file.
    offset : int
        The offset to start reading from (default: None).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the trial info in the same format as the Session trials table.
    list
        Timing data for each trial.
    """
    trials_table = []
    with open(jsonable_file) as f:
        if offset is not None:
            f.seek(offset, 0)
        for line in f:
            trials_table.append(json.loads(line))

    # pop-out the bpod data from the table
    bpod_data = []
    for td in trials_table:
        bpod_data.append(td.pop('behavior_data'))

    trials_table = pd.DataFrame(trials_table)
    return trials_table, bpod_data
