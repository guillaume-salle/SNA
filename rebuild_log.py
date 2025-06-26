import os
import wandb
import traceback
import hashlib
import argparse
import glob

from utils import (
    RunCompletionManager,
    load_and_process_config,
    config_to_stable_string,
    sanitize_for_wandb,
    expand_file_patterns,
)
from datasets import load_dataset_from_source

# Specify the WandB entity to use for logging.
WANDB_ENTITY = "USNA"


def rebuild_completion_log(problem_files, optimizer_files, num_seeds):
    """
    Rebuilds the completion log by fetching run data from W&B.
    It clears the existing log and repopulates it based on the specified configs.
    """
    api = wandb.Api()
    completion_manager = RunCompletionManager()

    # Clear the existing log file by opening in 'w' mode
    try:
        with open(completion_manager.log_filepath, "w") as f:
            f.write("")
        print(f"--- Cleared existing completion log: {completion_manager.log_filepath} ---")
    except Exception as e:
        print(f"Warning: Could not clear log file. Error: {e}")

    for p_path in problem_files:
        p_name = os.path.basename(p_path).replace(".yaml", "")
        print(f"\n--- Processing problem: {p_name} ---")
        p_config = load_and_process_config(p_path, {})
        dataset_params = p_config.get("dataset_params", {})

        # --- Infer param_dim (same logic as main.py) ---
        param_dim = dataset_params.get("param_dim") or p_config.get("param_dim")
        if not param_dim and "true_theta" in dataset_params:
            param_dim = len(dataset_params["true_theta"])
        elif not param_dim:
            if "synthetic" in p_name:
                raise ValueError(f"Synthetic problem '{p_name}' needs param_dim or true_theta.")
            dataset_name = p_config.get("dataset")
            if not dataset_name:
                raise ValueError(f"Problem config '{p_name}' is missing 'dataset' field.")
            loaded_data = load_dataset_from_source(dataset_name=dataset_name, random_state=0, **dataset_params)
            num_features = loaded_data["number_features"]
            bias = p_config.get("model_params", {}).get("bias", False)
            param_dim = num_features + 1 if bias else num_features

        context = {"d": param_dim, "n": dataset_params.get("n_dataset")}
        sanitized_p_name = sanitize_for_wandb(p_name)
        project_path = f"{WANDB_ENTITY}/{sanitized_p_name}"

        for o_path in optimizer_files:
            o_name = os.path.basename(o_path).replace(".yaml", "")
            o_config = load_and_process_config(o_path, context)
            sanitized_o_name = sanitize_for_wandb(o_name)

            print(f"  Checking optimizer: {o_name}")

            # --- Find runs on W&B ---
            runs_found = []
            try:
                # Filter by group (optimizer name) to narrow down the search
                runs = api.runs(project_path, filters={"group": sanitized_o_name})
                runs_found = list(runs)
                print(f"    Found {len(runs_found)} runs in group '{sanitized_o_name}'. Matching seeds...")
            except Exception as e:
                print(f"    ! ERROR querying W&B for project {project_path} and group {sanitized_o_name}. Error: {e}")
                continue

            # Create a map of seed -> run for easy lookup
            runs_by_seed = {run.config.get("seed"): run for run in runs_found if "seed" in run.config}

            for seed in range(num_seeds):
                run_description_for_log = f"{p_name}/{o_name}/seed_{seed}"

                if seed in runs_by_seed:
                    run = runs_by_seed[seed]
                    # Recreate the unique_run_id to log it correctly
                    current_run_config = {**p_config, **o_config, "seed": seed}
                    completion_id_stable_string = config_to_stable_string(current_run_config)
                    completion_id_hash = hashlib.md5(completion_id_stable_string.encode()).hexdigest()[:8]
                    unique_run_id = f"{sanitized_o_name}_{completion_id_hash}_{seed}"

                    # Find the local run directory by matching the unique W&B run ID
                    run_id = run.id
                    local_run_dirs = glob.glob(f"wandb/run-*-{run_id}")

                    if len(local_run_dirs) == 1:
                        local_dir = local_run_dirs[0]
                        completion_manager.log_run_completion(unique_run_id, run.id, local_dir, p_name)
                        print(f"    + Re-logged completed run: {run_description_for_log} (W&B ID: {run.id})")
                    elif not local_run_dirs:
                        print(
                            f"    ! Could not find local directory for run {run.id}. Run might not have been executed locally. Skipping."
                        )
                    else:
                        print(
                            f"    ! Found multiple possible local directories for run {run_id}: {local_run_dirs}. Skipping."
                        )

                else:
                    print(f"    - Missing run on W&B: {run_description_for_log}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild completion log from W&B.")
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
