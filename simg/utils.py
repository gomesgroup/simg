from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch

from simg.config import ATOMS


def one_hot_encoding(x: str, allowable_set: List[str], encode_unknown: Optional[bool] = False) -> List[bool]:
    """One-hot encodes a value.

    Parameters
    ----------
    x: str
        Value to one-hot encode.
    allowable_set: List[str]
        List of allowable values.
    encode_unknown: bool, optional
        Whether to encode unknown values.

    Returns
    -------
    List[bool]
        One-hot encoded value.
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))


def atom_type_one_hot(atom: str, allowable_set: Optional[List[str]] = ATOMS,
                      encode_unknown: Optional[bool] = False) -> List[bool]:
    """One-hot encodes an atom type.

    Parameters
    ----------
    atom: str
        Atom type to one-hot encode.
    allowable_set: List[str], optional
        List of allowable values.
    encode_unknown: bool, optional
        Whether to encode unknown values.

    Returns
    -------
    List[bool]
        One-hot encoded atom type.
    """
    return one_hot_encoding(atom, allowable_set, encode_unknown)


def process_cache(indexes, y, y_hat):
    y_slice = y[indexes]
    y_hat_slice = y_hat[indexes]

    distances = (y_slice - y_hat_slice[:, None]) ** 2
    distances = distances.sum(axis=2)
    y_hat_index, y_index = linear_sum_assignment(distances.detach().cpu().numpy())

    return indexes[y_index]


def get_matching(groups: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> List[int]:
    matches = []

    prev_value = -1
    cache = []

    for i, value in enumerate(np.append(groups, -1)):
        if value != prev_value and prev_value != -1:
            if len(cache) == 1:
                matches.append(cache[0])
            else:
                matches += list(process_cache(np.array(cache), y, y_hat))
            cache = []

        if value != -1:
            cache.append(i)

        if value == -1:
            matches.append(i)

        prev_value = value

    return matches[:-1]


def convert_batch_to_mask(batch: torch.LongTensor):
    out = []

    last_value = -1
    cache = 0

    for i in range(len(batch)):
        if batch[i] != last_value:
            if cache > 0:
                out.append(torch.ones((cache, cache), dtype=torch.float32))

            cache = 0

        cache += 1
        last_value = batch[i]

    out.append(torch.ones((cache, cache), dtype=torch.float32))
    out = torch.block_diag(*out)

    return out