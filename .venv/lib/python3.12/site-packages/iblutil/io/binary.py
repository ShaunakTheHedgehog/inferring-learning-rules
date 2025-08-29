from io import IOBase
from os import PathLike
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
import pandas as pd


def load_as_dataframe(
    filepath_bin: PathLike | str,
    dtype: np.dtype,
    count: int = -1,
    offset: int = 0,
) -> pd.DataFrame:
    """
    Load a binary file into a pandas DataFrame using a specified NumPy structured data type.

    Parameters
    ----------
    filepath_bin : Path or str
        The path to the binary file to be loaded. Can be a string or a Path object.
    dtype : np.dtype
        A NumPy structured data type that defines the format of the data in the binary file.
        Must be a structured datatype with fields.
    count : int, optional
        The number of items to read from the binary file. Default is -1, which means all items.
    offset : int, optional
        The number of bytes to skip at the beginning of the file before reading data. Default is 0.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data read from the binary file.

    Raises
    ------
    FileNotFoundError
        If the specified binary file does not exist.
    IsADirectoryError
        If the specified path is a directory instead of a file.
    ValueError
        If the provided dtype is not a NumPy structured datatype.
    """
    filepath_bin = Path(filepath_bin)
    if not filepath_bin.exists():
        raise FileNotFoundError(filepath_bin)
    if filepath_bin.is_dir():
        raise IsADirectoryError(filepath_bin)
    if not isinstance(dtype, np.dtype) or not hasattr(dtype, 'fields') or dtype.fields is None:
        raise ValueError('dtype must be a NumPy structured datatype')
    structured_array = np.fromfile(file=filepath_bin, dtype=dtype, count=count, offset=offset)
    return pd.DataFrame(structured_array)


def convert_to_parquet(
    filepath_bin: PathLike | str,
    dtype: np.dtype,
    delete_bin_file: bool = False,
) -> Path:
    """
    Convert a binary file to a Parquet file using a specified NumPy structured data type.

    Parameters
    ----------
    filepath_bin : Path or str
        The path to the binary file to be converted. Can be a string or a Path object.
    dtype : np.dtype
        A NumPy structured data type that defines the format of the data in the binary file.
        Must be a structured datatype with fields.
    delete_bin_file : bool, optional
        If True, the original binary file will be deleted after conversion. Default is False.

    Returns
    -------
    Path
        The path to the newly created Parquet file. The new filename will be constructed from
        the original filename and a '.pqt' suffix.

    Raises
    ------
    FileNotFoundError
        If the specified binary file does not exist.
    FileExistsError
        If the output file already exists.
    IsADirectoryError
        If the specified path is a directory instead of a file.
    ValueError
        If the provided dtype is not a NumPy structured datatype.
    """
    dataframe = load_as_dataframe(filepath_bin=filepath_bin, dtype=dtype)
    filepath_bin = Path(filepath_bin)
    filepath_pqt = filepath_bin.with_suffix('.pqt')
    if filepath_pqt.exists():
        raise FileExistsError(filepath_pqt)
    dataframe.to_parquet(filepath_pqt)
    if delete_bin_file:
        filepath_bin.unlink()
    return filepath_pqt


def write_array(fid: BinaryIO | str | PathLike, array: npt.ArrayLike, dtype: np.dtype):
    """
    Write a structured NumPy array to a binary file.

    Parameters
    ----------
    fid : bytes, str, IO
        The file path or file-like object where the structured array will be written.
    array : npt.ArrayLike
        The input array to be written. It must have a maximum of two dimensions,
        and the last dimension must match the number of fields in the provided dtype.
    dtype : np.dtype
        A structured NumPy datatype that defines the fields of the array.
        It must be a valid structured dtype with fields.

    Raises
    ------
    ValueError
        If `dtype` is not a structured NumPy datatype.
        If the input `array` has more than two dimensions.
        If the last dimension of `array` does not match the number of fields in `dtype`.
    FileExistsError
        If `fid` represents a Path and the respective file already exists.
    TypeError
        If `fid` is not a stream and cannot be converted to a Path.
    """
    if not isinstance(dtype, np.dtype) or not hasattr(dtype, 'fields') or dtype.fields is None:
        raise ValueError("'dtype' must be a structured NumPy datatype")
    array = np.array(array)
    if array.ndim > 2:
        raise ValueError('The array must have a maximum of two dimensions.')
    if array.shape[-1] != len(dtype.fields):
        raise ValueError("The array's last dimension must match the number of fields in 'dtype'.")
    if not isinstance(fid, IOBase) and Path(fid).exists():  # type: ignore
        raise FileExistsError(fid)
    structured_data = np.rec.fromrecords(array).astype(dtype)
    structured_data.tofile(fid)
