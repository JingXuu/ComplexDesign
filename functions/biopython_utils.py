import os
import math
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, DSSP, Selection, Polypeptide, PDBIO, Select, Chain, Superimposer
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa

def normalize_target_chains_to_ac(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_path)
    model = structure[0]

    # Use standard amino-acid residues only to identify protein chains.
    # Do not use this criterion later to filter output atoms.
    protein_chain_ids = []
    for chain in model:
        if any(is_aa(residue, standard=True) for residue in chain):
            protein_chain_ids.append(chain.id)

    if len(protein_chain_ids) != 2:
        raise ValueError(
            f"Expected exactly 2 protein chains in {pdb_path}, "
            f"found {len(protein_chain_ids)}: {protein_chain_ids}"
        )

    selected_chains = protein_chain_ids
    mapping = {
        selected_chains[0]: "A",
        selected_chains[1]: "C",
    }

    all_chain_ids = [chain.id for chain in model]

    has_conect = False
    with open(pdb_path, "r") as infile:
        for line in infile:
            if line.startswith("CONECT"):
                has_conect = True
                break

    # Reuse the original file only when it already contains exactly the two
    # canonical target chains and has no CONECT records to remove.
    if (
        selected_chains == ["A", "C"]
        and len(all_chain_ids) == 2
        and set(all_chain_ids) == {"A", "C"}
        and not has_conect
    ):
        return pdb_path, "A,C"

    root, ext = os.path.splitext(pdb_path)
    normalized_pdb_path = f"{root}_AC{ext or '.pdb'}"

    print(f"Input chain {selected_chains[0]} -> target chain A")
    print(f"Input chain {selected_chains[1]} -> target chain C")
    print(f"Writing normalized target PDB: {normalized_pdb_path}")

    coordinate_records = {"ATOM  ", "HETATM", "ANISOU"}
    last_written_chain = None

    with open(pdb_path, "r") as infile, open(normalized_pdb_path, "w") as outfile:
        for line in infile:
            record = line[:6]

            if record in coordinate_records:
                chain_id = line[21]
                if chain_id in mapping:
                    remapped_chain = mapping[chain_id]
                    outfile.write(line[:21] + remapped_chain + line[22:])
                    last_written_chain = chain_id
                continue

            if record == "TER   ":
                chain_id = line[21]

                if chain_id in mapping:
                    remapped_chain = mapping[chain_id]
                    outfile.write(line[:21] + remapped_chain + line[22:])
                    last_written_chain = None
                elif last_written_chain in mapping:
                    remapped_chain = mapping[last_written_chain]
                    outfile.write(line[:21] + remapped_chain + line[22:])
                    last_written_chain = None
                continue

            # Drop CONECT because the normalized PDB keeps only the two selected
            # target chains, so original connectivity records may become stale.
            if record == "CONECT":
                continue

            # Drop other records, including SEQRES/HELIX/SHEET, because they may
            # contain stale chain IDs after remapping. The design pipeline uses
            # coordinate records from the two target chains.
            if record == "END   ":
                continue

        outfile.write("END\n")

    return normalized_pdb_path, "A,C"

# analyze sequence composition of design
def validate_design_sequence(sequence, num_clashes, advanced_settings):
    note_array = []

    # Check if protein contains clashes after relaxation
    if num_clashes > 0:
        note_array.append('Relaxed structure contains clashes.')

    # Check if the sequence contains disallowed amino acids
    if advanced_settings["omit_AAs"]:
        restricted_AAs = advanced_settings["omit_AAs"].split(',')
        for restricted_AA in restricted_AAs:
            if restricted_AA in sequence:
                note_array.append('Contains: '+restricted_AA+'!')

    # Analyze the protein
    analysis = ProteinAnalysis(sequence)

    # Calculate the reduced extinction coefficient per 1% solution
    extinction_coefficient_reduced = analysis.molar_extinction_coefficient()[0]
    molecular_weight = round(analysis.molecular_weight() / 1000, 2)
    extinction_coefficient_reduced_1 = round(extinction_coefficient_reduced / molecular_weight * 0.01, 2)

    # Check if the absorption is high enough
    if extinction_coefficient_reduced_1 <= 2:
        note_array.append(f'Absorption value is {extinction_coefficient_reduced_1}, consider adding tryptophane to design.')

    # Join the notes into a single string
    notes = ' '.join(note_array)

    return notes

# temporary function, calculate RMSD of input PDB and trajectory target
def target_pdb_rmsd(trajectory_pdb, starting_pdb, chain_ids_string):
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure('trajectory', trajectory_pdb)
    structure_starting = parser.get_structure('starting', starting_pdb)
    
    # Extract chain A from trajectory_pdb
    chain_trajectory = structure_trajectory[0]['A']
    
    # Extract the specified chains from starting_pdb
    chain_ids = chain_ids_string.split(',')
    residues_starting = []
    for chain_id in chain_ids:
        chain_id = chain_id.strip()
        chain = structure_starting[0][chain_id]
        for residue in chain:
            if is_aa(residue, standard=True):
                residues_starting.append(residue)
    
    # Extract residues from chain A in trajectory_pdb
    residues_trajectory = [residue for residue in chain_trajectory if is_aa(residue, standard=True)]
    
    # Ensure that both structures have the same number of residues
    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]
    
    # Collect CA atoms from the two sets of residues
    atoms_starting = [residue['CA'] for residue in residues_starting if 'CA' in residue]
    atoms_trajectory = [residue['CA'] for residue in residues_trajectory if 'CA' in residue]
    
    # Calculate RMSD using structural alignment
    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    rmsd = sup.rms
    
    return round(rmsd, 2)

# temporary function, calculate RMSD of input PDB and trajectory target
def target_pdb_rmsd_multi(trajectory_pdb, starting_pdb, chain_ids_string):
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure('trajectory', trajectory_pdb)
    structure_starting = parser.get_structure('starting', starting_pdb)
    
    model_trajectory = structure_trajectory[0]
    model_start = structure_starting[0] 

    chain_ids = [ch.strip() for ch in chain_ids_string.split(',') if ch.strip()]

    residues_starting = []
    residues_trajectory = []

    for chain_id in chain_ids:                                
        if chain_id not in model_start or chain_id not in model_trajectory:  
            continue                                          

        chain_start = model_start[chain_id]                   
        chain_trajectory = model_trajectory[chain_id]                    

        res_start = [res for res in chain_start if is_aa(res, standard=True)]  
        res_trajectory = [res for res in chain_trajectory if is_aa(res, standard=True)]  

        if not res_start or not res_trajectory:                     
            continue                                          

        min_length = min(len(res_start), len(res_trajectory))       
        res_start  = res_start[:min_length]                   
        res_trajectory  = res_trajectory[:min_length]                    

        residues_starting.extend(res_start)                   
        residues_trajectory.extend(res_trajectory)                  

    if not residues_starting or not residues_trajectory:      
        return 0.0  

    # Ensure that both structures have the same number of residues
    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]
    
    # Collect CA atoms from the two sets of residues
    atoms_starting = [residue['CA'] for residue in residues_starting if 'CA' in residue]
    atoms_trajectory = [residue['CA'] for residue in residues_trajectory if 'CA' in residue]
    
    # Calculate RMSD using structural alignment
    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    rmsd = sup.rms
    
    return round(rmsd, 2)

# detect C alpha clashes for deformed trajectories
def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms = []
    atom_info = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':
                        continue
                    if only_ca and atom.get_name() != 'CA':
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((chain.id, residue.id[1], atom.get_name(), atom.coord))

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for (i, j) in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # Exclude clashes within the same residue
        if chain_i == chain_j and res_i == res_j:
            continue

        # Exclude directly sequential residues in the same chain for all atoms
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # If calculating sidechain clashes, only consider clashes between different chains
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)

three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# identify interacting residues at the binder interface
def hotspot_residues_multi(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0, target_chains=None, return_per_chain=False):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)
    model = structure[0]

    if target_chains is None:
        chains = ["A"]
    elif isinstance(target_chains, str):
        chains = [ch.strip() for ch in target_chains.split(",") if ch.strip()]
    else:
        chains = list(target_chains)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[ 0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}
    per_chain_contacts = {ch: set() for ch in chains}

    for t_chain in chains:
        if t_chain not in model:
            continue

        # Get atoms and coords for this target chain
        target_atoms = Selection.unfold_entities(model[t_chain], 'A')
        target_coords = np.array([atom.coord for atom in target_atoms])

        # Build KD tree for this target chain
        target_tree = cKDTree(target_coords)

        # Query the tree for pairs of atoms within the distance cutoff
        pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

        # Process each binder atom's interactions
        for binder_idx, close_indices in enumerate(pairs):
            binder_residue = binder_atoms[binder_idx].get_parent()
            binder_resname = binder_residue.get_resname()

            # Convert three-letter code to single-letter code using the manual dictionary
            if binder_resname in three_to_one_map:
                aa_single_letter = three_to_one_map[binder_resname]
                for close_idx in close_indices:
                    target_residue = target_atoms[close_idx].get_parent()
                    res_id = binder_residue.id[1]
                    interacting_residues[res_id] = aa_single_letter
                    per_chain_contacts[t_chain].add(res_id)

    if return_per_chain:
        return interacting_residues, per_chain_contacts
    else:
        return interacting_residues

# identify interacting residues at the binder interface
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[ 0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0]['A'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues

# calculate secondary structure percentage of design
def calc_ss_percentage(pdb_file, advanced_settings, chain_id="B", atom_distance_cutoff=4.0):
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues = set(hotspot_residues(pdb_file, chain_id, atom_distance_cutoff).keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]
            ss_type = 'loop'
            if ss in ['H', 'G', 'I']:
                ss_type = 'helix'
            elif ss == 'E':
                ss_type = 'sheet'

            ss_counts[ss_type] += 1

            if ss_type != 'loop':
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

# calculate secondary structure percentage of design
def calc_ss_percentage_multi(pdb_file, advanced_settings, chain_id="B", atom_distance_cutoff=4.0):
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues, _ = hotspot_residues_multi(
        pdb_file,
        binder_chain=chain_id,
        atom_distance_cutoff=atom_distance_cutoff,
        target_chains="A,C",
        return_per_chain=True,
    )
    interacting_residues = set(interacting_residues.keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]
            ss_type = 'loop'
            if ss in ['H', 'G', 'I']:
                ss_type = 'helix'
            elif ss == 'E':
                ss_type = 'sheet'

            ss_counts[ss_type] += 1

            if ss_type != 'loop':
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

def calculate_percentages(total, helix, sheet):
    helix_percentage = round((helix / total) * 100,2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100,2) if total > 0 else 0
    loop_percentage = round(((total - helix - sheet) / total) * 100,2) if total > 0 else 0

    return helix_percentage, sheet_percentage, loop_percentage
