import os
import wandb
import traceback
import hashlib
import argparse

from utils import (
    RunCompletionManager,
    load_and_process_config,
    config_to_stable_string,
    sanitize_for_wandb,
    expand_file_patterns,
    run_visualizations,
    find_best_lr,
    load_config,
    process_config_values,
)
from run import run_experiment
from datasets import load_dataset_from_source


# Specify the WandB entity to use for logging.
# Set to your username or a team name (e.g., "USNA").
# If set to None, the script will automatically use your default entity.
WANDB_ENTITY = "USNA"


def main():
    """
    Main function to run experiments, check completion, and visualize results.
    """
    parser = argparse.ArgumentParser(description="Run optimization experiments.")
    parser.add_argument(
        "-p",
        "--problems",
        nargs="+",
        required=True,
        help="Path(s) to YAML problem configuration files. Wildcards are supported.",
    )
    parser.add_argument(
        "-o",
        "--optimizers",
        nargs="+",
        required=True,
        help="Path(s) to YAML optimizer configuration files. Wildcards are supported.",
    )
    parser.add_argument(
        "-N",
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds to run for each experiment.",
    )
    parser.add_argument(
        "--check-runs",
        action="store_true",
        help="Only check for completed runs and report missing ones without running them.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visuals (table or plot) if all experiments are complete. Does not run experiments.",
    )
    parser.add_argument(
        "--find-lr",
        action="store_true",
        help="Find a good learning rate for a given problem and optimizer, then exit.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force experiments to re-run even if they are marked as complete in the log.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate a LaTeX-formatted table instead of the standard console table.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run wandb in offline mode to accelerate experiments.",
    )
    args = parser.parse_args()

    problem_files = expand_file_patterns(args.problems)
    optimizer_files = expand_file_patterns(args.optimizers)

    if not problem_files:
        print("No problem configuration files found from the provided patterns.")
        return
    if not optimizer_files:
        print("No optimizer configuration files found from the provided patterns.")
        return

    # --- Set up completion manager ---
    completion_manager = RunCompletionManager()
    completion_log_data = completion_manager.get_log_data()
    runs_to_fetch = []
    missing_runs = []
    all_p_configs = {}

    for p_path in problem_files:
        p_name = os.path.basename(p_path).replace(".yaml", "")
        print(f"\n{'='*30} Processing Problem: {p_name} {'='*30}")

        # --- Config Processing ---
        # 1. Load the raw config first to find the param_dim.
        raw_p_config = load_config(p_path)
        dataset_params = raw_p_config.get("dataset_params", {})

        # 2. Enforce that param_dim ('d') is explicitly defined.
        param_dim = dataset_params.get("param_dim")
        if param_dim is None:
            if "synthetic" in p_name and "true_theta" in dataset_params:
                param_dim = len(dataset_params["true_theta"])
            else:
                raise ValueError(
                    f"Configuration Error in '{p_path}': 'param_dim' must be explicitly "
                    "defined in the 'dataset_params' section."
                )

        # 3. Create context and process the final config with expressions.
        context = {"d": param_dim, "n": dataset_params.get("n_dataset")}
        p_config = process_config_values(raw_p_config, context)
        all_p_configs[p_name] = p_config

        if args.find_lr:
            print(f"\n--- Running LR Finder for Problem '{p_name}' ---")
            find_best_lr(p_config, seed=0)
            continue  # Skip to the next problem

        for o_path in optimizer_files:
            o_name = os.path.basename(o_path).replace(".yaml", "")
            o_config = load_and_process_config(o_path, context)

            for i in range(args.num_seeds):
                seed = i

                # --- Create unique run ID and check for completion ---
                current_run_config = {**p_config, **o_config, "seed": seed}
                completion_id_stable_string = config_to_stable_string(current_run_config)
                completion_id_hash = hashlib.md5(completion_id_stable_string.encode()).hexdigest()[:8]
                sanitized_o_name = sanitize_for_wandb(o_name)
                unique_run_id = f"{sanitized_o_name}_{completion_id_hash}_{seed}"

                completed_run_data = completion_log_data.get(p_name, {}).get(unique_run_id)

                if completed_run_data and completed_run_data.get("local_dir") and not args.rerun:
                    # Run is complete and has a valid local directory
                    run_info = {
                        "p_name": p_name,
                        "o_name": o_name,
                        "seed": seed,
                        "wandb_id": completed_run_data["wandb_id"],
                        "local_dir": completed_run_data["local_dir"],
                    }
                    runs_to_fetch.append(run_info)
                    continue

                # --- At this point, the run is MISSING or has an invalid log entry ---

                if (args.check_runs or args.visualize) and not args.rerun:
                    # In a "don't run" mode, just record it and move on.
                    missing_runs.append(f"Problem='{p_name}', Optimizer='{o_name}', Seed={seed}")
                    continue

                # --- EXECUTION MODE for a missing run ---
                print(f"--- Running missing experiment: Problem='{p_name}', Optimizer='{o_name}', Seed={seed} ---")
                wandb_run = None
                success = False
                try:
                    entity_to_use = WANDB_ENTITY
                    if not entity_to_use:
                        try:
                            entity_to_use = wandb.api.default_entity
                            print(f"--- Using default wandb entity: '{entity_to_use}' ---")
                        except AttributeError:
                            print(
                                "--- Could not determine default wandb entity. You may need to log in. Letting wandb decide. ---"
                            )
                            entity_to_use = None
                    sanitized_p_name = sanitize_for_wandb(p_name)
                    group_name = sanitized_o_name
                    wandb_run_name = f"{sanitized_o_name}"

                    print(f"--- Starting run: {wandb_run_name} (Project: {sanitized_p_name}, Group: {group_name}) ---")

                    wandb_run = wandb.init(
                        entity=entity_to_use,
                        project=sanitized_p_name,
                        config=current_run_config,
                        group=group_name,
                        name=wandb_run_name,
                        mode="offline" if args.offline else "online",
                    )
                    actual_wandb_id = wandb_run.id
                    local_dir = wandb_run.dir
                    run_experiment(p_config, o_config, seed=i)

                    success = True
                    completion_manager.log_run_completion(unique_run_id, actual_wandb_id, local_dir, p_name)
                    print(f"--- Finished and logged run (ID: {unique_run_id} / wandb: {actual_wandb_id}) ---")

                    # Add the newly completed run to our list for potential visualization
                    if args.visualize:
                        run_info = {
                            "p_name": p_name,
                            "o_name": o_name,
                            "seed": seed,
                            "wandb_id": actual_wandb_id,
                            "local_dir": local_dir,
                        }
                        runs_to_fetch.append(run_info)
                except Exception as e:
                    print(f"!!! ERROR during execution for {o_name}, seed {seed}: {e} !!!")
                    traceback.print_exc()
                    raise
                finally:
                    if wandb_run is not None:
                        # Force a save of the run's data before finishing.
                        # This can help ensure data is written, especially in offline mode.
                        wandb_run.save()
                        exit_code = 0 if success else 1
                        wandb.finish(exit_code=exit_code)
                        print(f"--- WandB run finished for {o_name} (Exit code: {exit_code}) ---")

    # --- Post-Loop Actions ---
    if args.find_lr:
        print("\n--- LR Finder process complete. ---")
        return

    if missing_runs:
        # This block is only reached if --check-runs or --visualize was used and runs were missing.
        print("\n--- Missing Experiments ---")
        print("The following experiment runs were not found in the completion log:")
        for desc in missing_runs:
            print(f"  - {desc}")
        if args.visualize:
            print("\nCannot generate visuals because some experiments are missing. Please run them first.")
        return

    # If we get here, all specified experiments are now complete.
    if args.visualize:
        print("\n--- All experiments are completed. Proceeding with visualization... ---")
        run_visualizations(runs_to_fetch, problem_files, all_p_configs, latex=args.latex, entity=WANDB_ENTITY)
    elif args.check_runs:
        print("\n--- All experiments are completed. ---")
    else:
        # This is the default case (no flags) where all runs were either already complete or were just run.
        print("\n--- All necessary experiments are now complete. ---")

    if args.offline:
        print("\n" + "=" * 60)
        print("--- To sync your offline runs with the wandb server, run: ---")
        print("wandb sync --sync-all")
        print("=" * 60)


if __name__ == "__main__":
    main()
