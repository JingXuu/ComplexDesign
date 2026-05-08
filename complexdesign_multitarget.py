from functions import *
from ColabDesign.colabdesign.shared.utils import copy_dict

def main():
    check_jax_gpu()

    parser = argparse.ArgumentParser(description='Script to run ComplexDesign design pipeline.')

    parser.add_argument('--settings', '-s', type=str, required=True,
                        help='Path to the basic settings.json file. Required.')
    parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                        help='Path to the filters.json file used to filter design. If not provided, default will be used.')
    parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_complexdesign_multitarget.json',
                        help='Path to the advanced.json file with additional design settings. If not provided, default will be used.')

    args = parser.parse_args()

    settings_path, filters_path, advanced_path = perform_input_check(args)

    target_settings, advanced_settings, _ = load_json_settings(settings_path, filters_path, advanced_path)
    normalized_pdb_path, normalized_chains = normalize_target_chains_to_ac(
        target_settings["starting_pdb"],
    )
    target_settings["starting_pdb"] = normalized_pdb_path
    target_settings["chains"] = normalized_chains

    settings_file = os.path.basename(settings_path).split('.')[0]
    filters_file = os.path.basename(filters_path).split('.')[0]
    advanced_file = os.path.basename(advanced_path).split('.')[0]

    design_models, _, _ = load_af2_models(advanced_settings["use_multimer_design"])

    complexdesign_folder = os.path.dirname(os.path.realpath(__file__))
    advanced_settings = perform_advanced_settings_check(advanced_settings, complexdesign_folder)

    design_paths = generate_directories(target_settings["design_path"])



    trajectory_labels = get_trajectory_labels()

    trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
    failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

    create_dataframe(trajectory_csv, trajectory_labels)
    generate_filter_pass_csv(failure_csv, args.filters)

    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
    print(f"Running binder design for target {settings_file}")
    print(f"Design settings used: {advanced_file}")
    print(f"Filtering designs based on {filters_file}")


    # initialise counters
    script_start_time = time.time()
    trajectory_n = 0
    accept_pdbs = 0
    accepted_designs = 0
    target_final_designs = int(target_settings["number_of_final_designs"])
    min_len = int(min(target_settings["lengths"]))
    max_len = int(max(target_settings["lengths"]))

    while accepted_designs < target_final_designs:

        trajectory_start_time = time.time()

        seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
        length = int(np.random.randint(min_len, max_len + 1))


        helicity_value = load_helicity(advanced_settings)

        design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
        trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
        trajectory_exists = any(os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs)

        if not trajectory_exists:
            print("Starting trajectory: "+design_name)

            trajectory = binder_hallucination_multi_target(design_name, target_settings["starting_pdb"], target_settings["chains"],
                                                target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                                design_models, advanced_settings, design_paths, failure_csv)
            trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"]) # contains plddt, ptm, i_ptm, pae, i_pae


            trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}

            trajectory_time = time.time() - trajectory_start_time
            trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
            print("Starting trajectory took: "+trajectory_time_text)
            print("")

            if trajectory.aux["log"]["terminate"] == "":

                binder_chain = "B"

                trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
                chain_res_count = count_chain_residues(target_settings["starting_pdb"])
                restore_AC_chains(trajectory_pdb, trajectory_pdb, chain_res_count)

                _, _, trajectory_interface_residues_all = score_interface_multi(trajectory_pdb, binder_chain)
                trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], 
                                    trajectory_metrics['plddt'], trajectory_metrics['ptm'], trajectory_metrics['i_ptm'], trajectory_metrics['pae'], trajectory_metrics['i_pae'],
                                    trajectory_time_text, settings_file, filters_file, advanced_file]
                insert_data(trajectory_csv, trajectory_data)

                if advanced_settings["enable_mpnn"]:
                    mpnn_n = 1
                    design_start_time = time.time()

                    trajectory_interface_residues = ",".join(v for v in trajectory_interface_residues_all.values() if v)


                    mpnn_sequences = collect_mpnn_sequences_multitarget(
                        trajectory_pdb=trajectory_pdb,
                        binder_chain=binder_chain,
                        trajectory_interface_residues=trajectory_interface_residues,
                        advanced_settings=advanced_settings,
                        target_length=length,
                    )
    
                    if mpnn_sequences:
            
                        for mpnn_sequence in mpnn_sequences:

                            mpnn_design_name = design_name + "_mpnn" + str(mpnn_n)

                            if advanced_settings["save_mpnn_fasta"] is True:
                                save_fasta(mpnn_design_name, mpnn_sequence['seq'], design_paths)
                            mpnn_n += 1
                    design_time = time.time() - design_start_time
                    design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
                    print("Design "+str(len(mpnn_sequences))+" MPNN sequences of trajectory "+design_name+" took: "+design_time_text)
                    accept_pdbs += 1
                    accepted_designs += 1
                else:
                    accept_pdbs += 1
                    accepted_designs += 1


            trajectory_n += 1

            gc.collect()

    elapsed_time = time.time() - script_start_time
    elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"
    print("Finished all designs. Script execution for "+str(accept_pdbs)+" accepted trajectories, " +str(trajectory_n)+" total trajectories; took: "+elapsed_text)

if __name__ == "__main__":  
    main()
