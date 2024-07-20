import uuid
import os
from typing import List, Optional, Tuple

import numpy as np


def sort_nodes(connectivity):
    """Sort nodes in connectivity information.

    Parameters
    ----------
    connectivity: List[Tuple[int, int, int]]
        List of tuples containing atom indices, bond indices, and bond orders.

    Returns
    -------
    List[Tuple[int, int, int]]
        Ordered list of tuples containing atom indices, bond indices, and bond orders.
    """
    connectivity.sort(key=lambda x: (x[0], x[1]))
    return connectivity

def get_connectivity_info(xyz_data: Optional[List[str]] = None,
                          sdf_file: Optional[str] = None, indexing_function=sort_nodes) -> List[Tuple[int, int, int]]:
    """
    Get connectivity information from xyz or sdf file.

    Parameters
    ----------
    xyz_data: List[str]
        List of xyz lines.
    sdf_file: str
        Path to sdf file.

    Returns
    -------
    List[Tuple[int, int, int]]
        List of tuples containing atom indices, bond indices, and bond orders.
    """
    connectivity = []

    # Generate and read SDF from openbabel
    if sdf_file is None:
        unique_name = uuid.uuid4().hex
        xyz_temp_filename = unique_name + ".xyz"
        sdf_temp_filename = unique_name + ".sdf"

        # Write the xyz file
        with open(xyz_temp_filename, "w") as f:
            f.write(str(len(xyz_data)) + "\n\n")  # XYZ header
            for line in xyz_data:
                f.write(line)
        os.system(
            "obabel {0} -O {1} >/dev/null 2>&1".format(xyz_temp_filename, sdf_temp_filename)
        )

        with open(sdf_temp_filename) as f:
            sdf_lines = f.readlines()
    else:
        with open(sdf_file) as f:
            sdf_lines = f.readlines()

    # Get raw connectivity info from file
    sdf_lines = [line[:3] + ' ' + line[3:] for line in sdf_lines]
    sdf_lines = [x.strip() for x in sdf_lines][3:]
    sdf_header = sdf_lines[0].split()
    sdf_header = list(filter(lambda x: x, sdf_header))
    num_atoms, num_bonds = int(sdf_header[0]), int(sdf_header[1])
    raw_connectivity = sdf_lines[num_atoms + 1: -2]

    # Parse connectivity info
    for connection in raw_connectivity:
        A = connection.split(" ")
        A = [x for x in A if x != ""]
        if A[0] == 'M':
            continue
        # (source_node, target_node, bond_type)
        connectivity.append((int(A[0]) - 1, int(A[1]) - 1, int(A[2])))
    
    # Delete files after use
    if sdf_file is None:
        os.remove(xyz_temp_filename)
        os.remove(sdf_temp_filename)

    if indexing_function is not None:
        connectivity = indexing_function(connectivity)

    return connectivity

def get_atom_atom_edges(connectivity: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    """Get atom-atom edges from connectivity information.

    Parameters
    ----------
    connectivity: List[Tuple[int, int, int]]
        List of tuples containing atom indices, bond indices, and bond orders.

    Returns
    -------
    List[Tuple[int, int]]
        List of tuples containing atom indices.
    """
    edges = []
    for bond_idx, (atom_A_idx, atom_B_idx, bond_order) in enumerate(connectivity):
        edges.append([atom_A_idx, atom_B_idx])
        edges.append([atom_B_idx, atom_A_idx])
    return edges


def block_diagonal(a: np.ndarray, b: np.ndarray, n_features: int, default: Optional[float] = 0):
    """Constructs a block diagonal matrix from two matrices.

    Parameters
    ----------
    a: np.ndarray
        First matrix.
    b: np.ndarray
        Second matrix.
    n_features: int
        Number of features == number of columns in the output matrix.
    default: Optional[float]
        Default value for the output matrix.

    Returns
    -------
    np.ndarray
        Block diagonal matrix.
    """
    out = np.zeros((a.shape[0] + b.shape[0], n_features)) + default
    out[:a.shape[0], :a.shape[1]] = a

    if b.shape[0] != 0:
        out[a.shape[0]:, a.shape[1]:] = b

    return out
