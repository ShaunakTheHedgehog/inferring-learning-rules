import uuid
import json
import logging

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd


_logger = logging.getLogger('ibllib')


def load(filename):
    """
    Loads parquet file into pandas dataframe
    :param filename:
    :return:
    """
    table = pq.read_table(filename)
    try:
        metadata = json.loads(table.schema.metadata[b'one_metadata'])
    except KeyError:
        _logger.debug('No parquet metadata in %s', filename)
        metadata = {}
    df = table.to_pandas()
    return df, metadata


def save(filename, table, metadata=None):
    """
    Save pandas dataframe to parquet
    :param filename:
    :param table:
    :param metadata:
    :return:
    """
    # cf https://towardsdatascience.com/saving-metadata-with-dataframes-71f51f558d8e

    # from dataframe to parquet
    table = pa.Table.from_pandas(table)

    # Add user metadata
    table = table.replace_schema_metadata({'one_metadata': json.dumps(metadata or {}).encode(), **table.schema.metadata})

    # Save to parquet.
    pq.write_table(table, filename)


def uuid2np(eids_uuid):
    return np.asfortranarray(np.array([np.frombuffer(eid.bytes, dtype=np.int64) for eid in eids_uuid]))


def str2np(eids_str):
    """
    Converts uuid string or list of uuid strings to int64 numpy array with 2 cols
    Returns [0, 0] for None list entries
    """
    if isinstance(eids_str, str):
        eids_str = [eids_str]
    return uuid2np([uuid.UUID(eid) if eid else uuid.UUID('0' * 32) for eid in eids_str])


def np2uuid(eids_np):
    if isinstance(eids_np, pd.DataFrame) | isinstance(eids_np, pd.Series):
        eids_np = eids_np.to_numpy()
    if eids_np.ndim >= 2:
        return [uuid.UUID(bytes=npu.tobytes()) for npu in eids_np]
    else:
        return uuid.UUID(bytes=eids_np.tobytes())


def np2str(eids_np):
    eids = np2uuid(eids_np)
    eids = str(eids) if isinstance(eids, uuid.UUID) else [str(u) for u in np2uuid(eids_np)]
    return eids


def is_np_id(id):
    """
    The purpose of this is to correctly identify ids even as object arrays
    :param id:
    :return:
    """
    # TODO Document and test
    id = np.asarray(id)
    is_int = id.dtype == int or np.all(isinstance(x, int) for x in id)
    return id.shape[1] == 2 and is_int
