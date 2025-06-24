import argparse
import os
import wandb
import hashlib
from utils import (
    load_and_process_config,
    config_to_stable_string,
    sanitize_for_wandb,
    expand_file_patterns,
    RunCompletionManager,
)
from datasets import load_dataset_from_source

WANDB_ENTITY = "USNA"


def rebuild_completion_log(problem_files, optimizer_files, num_seeds):
    print("--- Attempting to rebuild completion log from W&B server ---")
    completion_manager = RunCompletionManager()
    api = wandb.Api()

    for p_path in problem_files:
        p_name = os.path.basename(p_path).replace(".yaml", "")
        print(f"\nProcessing Problem: {p_name}")
        try:
            p_config_base = load_and_process_config(p_path, {})

            # This logic is copied from main.py to ensure consistency
            p_config = p_config_base.copy()
            param_dim: int | None = None
            dataset_params = p_config.get("dataset_params", {})

            if "param_dim" in dataset_params:
                param_dim = int(dataset_params["param_dim"])
            elif "true_theta" in dataset_params:
                param_dim = len(dataset_params["true_theta"])
            else:
                if "synthetic" in p_name:
                    raise ValueError("param_dim must be specified for synthetic datasets")
                print(f"   Inferring param_dim for problem '{p_name}' by loading the dataset...")
                dataset_name = p_config.get("dataset")
                loaded_data = load_dataset_from_source(dataset_name=dataset_name, random_state=0, **dataset_params)
                num_features = loaded_data["number_features"]
                bias = p_config.get("model_params", {}).get("bias", False)
                param_dim = num_features + 1 if bias else num_features

            p_config["dataset_params"]["param_dim"] = param_dim
            context = {"d": param_dim, "n": dataset_params.get("n_dataset", int(1e6))}

            sanitized_p_name = sanitize_for_wandb(p_name)
            project_path = f"{WANDB_ENTITY}/{sanitized_p_name}"

            # --- New logic: Fetch all runs from the project first ---
            runs_in_project = {}
            runs_not_found_in_project = []
            try:
                print(f"   Fetching all runs from project: {project_path}")
                all_runs_for_project = api.runs(project_path)
                for run in all_runs_for_project:
                    runs_in_project[run.name] = run
                print(f"   ... found {len(runs_in_project)} runs.")
            except Exception as e:
                print(f"  ! ERROR querying W&B for project {project_path}. Maybe the project doesn't exist? Error: {e}")
                continue  # Skip to the next problem file

            for o_path in optimizer_files:
                o_name = os.path.basename(o_path).replace(".yaml", "")
                o_config = load_and_process_config(o_path, context)
                sanitized_o_name = sanitize_for_wandb(o_name)

                for seed in range(num_seeds):
                    wandb_run_name = f"{sanitized_o_name}"
                    run_description_for_log = f"{p_name}/{o_name}/seed_{seed}"

                    # Find the run in our fetched dictionary
                    if wandb_run_name in runs_in_project:
                        run = runs_in_project[wandb_run_name]
                        actual_wandb_id = run.id

                        # Recreate the unique_run_id to log it correctly
                        current_run_config = {**p_config, **o_config, "seed": seed}
                        completion_id_stable_string = config_to_stable_string(current_run_config)
                        completion_id_hash = hashlib.md5(completion_id_stable_string.encode()).hexdigest()[:8]
                        unique_run_id = f"{sanitized_o_name}_{completion_id_hash}_{seed}"

                        # Log it
                        completion_manager.log_run_completion(unique_run_id, actual_wandb_id, p_name)
                        print(f"  + Found and re-logged run: {run_description_for_log} (W&B ID: {actual_wandb_id})")
                    else:
                        runs_not_found_in_project.append(run_description_for_log)

            # After checking all optimizers, print debug info if any runs were missing
            if runs_not_found_in_project:
                print(f"\n  --- DEBUG: Runs not found for project '{p_name}' ---")
                for missing in runs_not_found_in_project:
                    print(f"    - Script was looking for: {missing}")
                print("\n    Available runs found on W&B server:")
                if not runs_in_project:
                    print("      (No runs were found in this project at all)")
                for available_run_name in runs_in_project.keys():
                    print(f"      - {available_run_name}")
                print("  ----------------------------------------------------")

        except Exception as e:
            print(f"!! Failed to process config for problem {p_name}. Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Rebuild completion log from W&B.")
    parser.add_argument(
        "-p",
        "--problems",
        nargs="+",
        required=True,
        help="Path(s) to YAML problem config files. Wildcards are supported.",
    )
    parser.add_argument(
        "-o",
        "--optimizers",
        nargs="+",
        required=True,
        help="Path(s) to YAML optimizer config files. Wildcards are supported.",
    )
    parser.add_argument("-N", "--num-seeds", type=int, default=1, help="Number of random seeds to check for.")
    args = parser.parse_args()

    problem_files = expand_file_patterns(args.problems)
    optimizer_files = expand_file_patterns(args.optimizers)

    if not problem_files:
        print("No problem configuration files found.")
        return
    if not optimizer_files:
        print("No optimizer configuration files found.")
        return

    rebuild_completion_log(problem_files, optimizer_files, args.num_seeds)
    print("\n--- Rebuilding process complete. ---")


if __name__ == "__main__":
    main()
