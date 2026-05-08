from functions import *
from ColabDesign.colabdesign.shared.utils import copy_dict



def main():
    check_jax_gpu()

    parser = argparse.ArgumentParser(
        description="Script to run ComplexDesign design pipeline."
    )

    parser.add_argument(
        "--settings",
        "-s",
        type=str,
        required=True,
        help="Path to the basic settings.json file. Required.",
    )
    parser.add_argument(
        "--filters",
        "-f",
        type=str,
        default="./settings_filters/default_filters.json",
        help="Path to the filters.json file used to filter design. If not provided, default will be used.",
    )
    parser.add_argument(
        "--advanced",
        "-a",
        type=str,
        default="./settings_advanced/default_4stage_complexdesign_unconditional.json",
        help="Path to the advanced.json file with additional design settings. If not provided, default will be used.",
    )

    args = parser.parse_args()

    settings_path, filters_path, advanced_path = perform_input_check(args)

    target_settings, advanced_settings, _ = load_json_settings(
        settings_path, filters_path, advanced_path
    )

    settings_file = os.path.basename(settings_path).split(".")[0]
    filters_file = os.path.basename(filters_path).split(".")[0]
    advanced_file = os.path.basename(advanced_path).split(".")[0]

    design_models, _, _ = load_af2_models(
        advanced_settings["use_multimer_design"]
    )

    complexdesign_folder = os.path.dirname(os.path.realpath(__file__))
    advanced_settings = perform_advanced_settings_check(
        advanced_settings, complexdesign_folder
    )

    design_paths = generate_directories(target_settings["design_path"])

    trajectory_labels = get_trajectory_labels()

    trajectory_csv = os.path.join(
        target_settings["design_path"], "trajectory_stats.csv"
    )
    failure_csv = os.path.join(target_settings["design_path"], "failure_csv.csv")

    create_dataframe(trajectory_csv, trajectory_labels)
    generate_filter_pass_csv(failure_csv, args.filters)

    pr.init(
        f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1'
    )
    print(f"Running binder design for target {settings_file}")
    print(f"Design settings used: {advanced_file}")
    print(f"Filtering designs based on {filters_file}")

    complex_lengths = target_settings.get("complex_lengths", None)
    if not complex_lengths or len(complex_lengths) == 0:
        raise ValueError(
            "settings.json must contain 'complex_lengths' as a 1D list, e.g. [120,160,90]."
        )
    lengths = [int(L) for L in complex_lengths]
    total_len = sum(lengths)
    print(f"Using complex lengths: {lengths} (total_len={total_len})")

    complex_name_prefix = target_settings.get("complex_name", "Complex")

    max_trajectories = int(advanced_settings.get("max_trajectories", 500))
    number_of_final_designs = int(target_settings.get("number_of_final_designs", 32))



    # initialise counters
    script_start_time = time.time()
    trajectory_n = 0
    accept_pdbs = 0
    accepted_designs = 0

    while trajectory_n < max_trajectories:
        trajectory_start_time = time.time()
        trajectory_n += 1

        trajectory_start_time = time.time()
        seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

        length_tag = "-".join(str(L) for L in lengths)
        design_name = f"{complex_name_prefix}_L{length_tag}_s{seed}"

        trajectory_dirs = [
            "Trajectory",
            "Trajectory/Relaxed",
            "Trajectory/LowConfidence",
            "Trajectory/Clashing",
        ]
        trajectory_exists = any(
            os.path.exists(
                os.path.join(design_paths[trajectory_dir], design_name + ".pdb")
            )
            for trajectory_dir in trajectory_dirs
        )

        if trajectory_exists:
            print(f"Trajectory {design_name} already exists, skipping.")
            continue

        print(f"Starting trajectory: {design_name} (lengths={lengths}, seed={seed})")

        trajectory = complex_hallucination_unconditional(
            design_name=design_name,
            lengths=lengths,
            seed=seed,
            design_models=design_models,
            advanced_settings=advanced_settings,
            design_paths=design_paths,
            failure_csv=failure_csv,
        )

        # time trajectory
        trajectory_time = time.time() - trajectory_start_time
        trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
        print("Starting trajectory took: " + trajectory_time_text)
        print("")

        trajectory_metrics = copy_dict(
            trajectory._tmp["best"]["aux"]["log"]
        )

        trajectory_metrics = {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in trajectory_metrics.items()
        }

        # Proceed if there is no trajectory termination signal
        if trajectory.aux["log"]["terminate"] == "":
            helicity_value = advanced_settings.get("helix_target", 0.0)
            hotspot_str = ""
            trajectory_data = [
                design_name,
                advanced_settings["design_algorithm"],
                total_len,
                seed,
                helicity_value,
                hotspot_str,
                trajectory_metrics.get("plddt", 0.0),
                trajectory_metrics.get("ptm", 0.0),
                trajectory_metrics.get("i_ptm", 0.0),
                trajectory_metrics.get("pae", 0.0),
                trajectory_metrics.get("i_pae", 0.0),
                trajectory_time_text,
                settings_file,
                filters_file,
                advanced_file,
            ]
            insert_data(trajectory_csv, trajectory_data)
            accept_pdbs += 1
            accepted_designs += 1

            if advanced_settings.get("enable_mpnn", False):
                mpnn_n = 1
                design_start_time = time.time()

                trajectory_pdb = os.path.join(
                    design_paths["Trajectory"], design_name + ".pdb"
                )

                mpnn_sequences = collect_mpnn_sequences_unconditional(
                    trajectory_pdb=trajectory_pdb,
                    lengths=lengths,
                    advanced_settings=advanced_settings,
                )

                if mpnn_sequences:
                    for mpnn_sequence in mpnn_sequences:
                        mpnn_design_name = f"{design_name}_mpnn{mpnn_n}"

                        if advanced_settings.get("save_mpnn_fasta", True):
                            save_multichain_fasta(
                                mpnn_design_name,
                                mpnn_sequence["parts"],
                                design_paths,
                            )
                        mpnn_n += 1

                    design_time = time.time() - design_start_time
                    design_time_text = (
                        f"{int(design_time // 3600)} hours, "
                        f"{int((design_time % 3600) // 60)} minutes, "
                        f"{int(design_time % 60)} seconds"
                    )
                    print(
                        f"Designed {len(mpnn_sequences)} MPNN sequences "
                        f"for trajectory {design_name} in {design_time_text}"
                    )

                else:
                    print(
                        f"[WARN] No valid MPNN sequences generated for {design_name} "
                        f"(filtered by omit_AAs / dedup)."
                    )
            if accepted_designs >= number_of_final_designs:
                break

        else:
            print(
                f"Trajectory {design_name} terminated with reason: "
                f"{trajectory.aux['log']['terminate']}"
            )

        gc.collect()
        

    elapsed_time = time.time() - script_start_time
    elapsed_text = (
        f"{int(elapsed_time // 3600)} hours, "
        f"{int((elapsed_time % 3600) // 60)} minutes, "
        f"{int(elapsed_time % 60)} seconds"
    )
    print(
        "Finished all designs. Script execution for "
        + str(accept_pdbs)
        + " accepted trajectories, "
        + str(trajectory_n)
        + " total; took: "
        + elapsed_text
    )

if __name__ == "__main__":
    main()
