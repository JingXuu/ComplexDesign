
def count_chain_residues(pdb_path):
    chain_res = {}

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21]
            res_seq = line[22:26]
            i_code = line[26]
            key = (chain_id, res_seq, i_code)

            if chain_id not in chain_res:
                chain_res[chain_id] = set()
            chain_res[chain_id].add(key)

    return {ch: len(res_set) for ch, res_set in chain_res.items()}

def restore_AC_chains(pdb_in, pdb_out, res_counts):
    len_A = res_counts.get("A")

    with open(pdb_in, "r") as fin:
        lines = fin.readlines()

    new_lines = []
    current_res_id = None  
    residue_counter = 0  
    c_res_counter = 0  

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            chain_id = line[21]

            if chain_id == "A":
                res_seq = line[22:26]
                i_code = line[26]
                res_id = (res_seq, i_code)

                if res_id != current_res_id:
                    current_res_id = res_id
                    residue_counter += 1
                    if residue_counter > len_A:
                        c_res_counter += 1

                if residue_counter <= len_A:
                    new_chain = "A"
                    new_line = line[:21] + new_chain + line[22:]
                else:
                    new_chain = "C"
                    new_resseq = f"{c_res_counter:4d}"   # 4 位右对齐
                    new_line = line[:21] + new_chain + new_resseq + line[26:]
                line = new_line

        new_lines.append(line)

    with open(pdb_out, "w") as fout:
        fout.writelines(new_lines)
