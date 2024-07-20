from typing import Optional, List, Tuple

import numpy as np

import scipy.sparse.csgraph as spgraph
from scipy.optimize import linear_sum_assignment

import torch
from torch_geometric.data import Data


def _get_graph_spectrum(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lap = spgraph.laplacian(adj)
    eigen_values, eigen_vectors = np.linalg.eigh(lap)
    eigen_values = eigen_values / np.max(eigen_values)
    return eigen_values, eigen_vectors


def _heat_wavelets(eigen_values: np.ndarray, eigen_vectors: np.ndarray,
                   scale: float) -> np.ndarray:
    kernel = np.diag(np.exp(-scale * eigen_values))
    wavelet_basis = eigen_vectors @ kernel @ eigen_vectors.T
    return wavelet_basis


def _characteristic_function(wavelet_basis: np.ndarray,
                             t: np.ndarray) -> np.ndarray:
    scaled_waves = np.einsum('ij,k->ijk', wavelet_basis, t)
    return np.mean(np.exp(scaled_waves * 1j), axis=0)


def get_graphwave_embeddings(adj: np.ndarray, ndim: Optional[int] = 72, step_size: Optional[float] = 4,
                             scales: Optional[List[float]] = [10, 50]) -> np.ndarray:
    n_scales = len(scales)
    sample_number = ndim // (2 * n_scales)
    t = np.arange(sample_number) * step_size
    eigen_values, eigen_vectors = _get_graph_spectrum(adj)
    embs = []
    for scale in scales:
        wavelet_basis = _heat_wavelets(eigen_values, eigen_vectors, scale)
        wavelets = _characteristic_function(wavelet_basis, t)
        embs.extend([wavelets.real, wavelets.imag])
    return np.hstack(embs)


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


def get_matching_not_sorted(groups: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    out = np.arange(len(y))

    for i in np.unique(groups):
        if i != -1:
            out[groups == i] = process_cache(out[groups == i], y, y_hat)

    return out


def get_group_idx(graph: Data):
    """For group find groups of permutation invariant nodes

    Parameters
    ----------
    graph: Data
        Graph to find groups of permutation invariant nodes

    Returns
    -------
    group_idx: np.ndarray
        Indexes of groups of permutation invariant nodes
    """

    out = np.zeros(len(graph.is_lp))
    out[~graph.is_lp.bool()] = -1

    for lp, atom in graph.edge_index[:, graph.is_lp[graph.edge_index[0]].bool()].numpy().T:
        out[lp] = atom

    return out


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
