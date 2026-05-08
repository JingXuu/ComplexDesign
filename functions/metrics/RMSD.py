import numpy as np
from typing import List
from biotite.structure.io import load_structure


##### copy parted from kabsch_algorithm ######


def kabsch_algorithm(P, Q):
    C_P = np.mean(P, axis=0)
    C_Q = np.mean(Q, axis=0)
    P_centered = P - C_P
    Q_centered = Q - C_Q
    H = np.dot(P_centered.T, Q_centered)

    try:
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
    except np.linalg.LinAlgError:
        print("Warning: SVD did not converge. Returning identity rotation.")
        R = np.eye(3)
    return R, C_P, C_Q


def calculate_rmsd(P, Q):
    diff = P - Q
    return np.sqrt(np.sum(diff * diff) / len(P))


def get_ca_coords_from_path(pdb_path: str, chain_list: List[str]):
    atom_array = load_structure(pdb_path)
    ca_atoms = atom_array[
        np.isin(atom_array.chain_id, chain_list) & (atom_array.atom_name == "CA")
    ]
    assert len(ca_atoms) > 0, f"No CA atoms found in chain {chain_list} of {pdb_path}"
    return ca_atoms.coord


def complex_ca_rmsd_from(
    path_1: str,
    path_2: str,
    chain_1_list: List[str] = ["A", "B"],
    chain_2_list: List[str] = ["A", "B"],
):
    """
    Input:
        path_1: str, path to the first pdb file
        path_2: str, path to the second pdb file
        chain_1_list: list of str, chain ids in the first pdb file
        chain_2_list: list of str, chain ids in the second pdb file
    Output:
        rmsd: float, CA RMSD between the two complexes
    """

    assert set(chain_1_list) == set(chain_2_list)

    coords1 = get_ca_coords_from_path(path_1, chain_1_list)
    coords2 = get_ca_coords_from_path(path_2, chain_2_list)

    assert len(coords1) == len(coords2)

    R, C_P, C_Q = kabsch_algorithm(coords1, coords2)
    coords2_aligned = np.dot(coords2 - C_Q, R) + C_P
    rmsd = calculate_rmsd(coords1, coords2_aligned)

    return round(rmsd, 3)


if __name__ == "__main__":
    path1 = "./model1.pdb"
    path2 = "./model2.pdb"
    rmsd = complex_ca_rmsd_from(
        path1,
        path2,
    )
    print(f"RMSD: {rmsd}")