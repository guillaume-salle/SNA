import os
import wandb
import traceback
import hashlib
import argparse

import torch

from utils import (
    RunCompletionManager,
    load_and_process_config,
    config_to_stable_string,
    sanitize_for_wandb,
    expand_file_patterns,
)
from run import run_experiment
from datasets import load_dataset_from_source


# Specify the WandB entity to use for logging.
# Set to your username or a team name (e.g., "USNA").
# If set to None, the script will automatically use your default entity.
WANDB_ENTITY = "USNA"


def main():
    """
    Main function to run experiments.
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
        "--log-file",
        type=str,
        default=RunCompletionManager.DEFAULT_LOG_FILE,
        help="Path to the log file for completed runs.",
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

    # Set up a completion manager to handle resumes
    completion_manager = RunCompletionManager(args.log_file)

    for p_path in problem_files:
        p_name = os.path.basename(p_path).replace(".yaml", "")
        p_config_base = load_and_process_config(p_path, {})

        # --- Create Context for Optimizer Configs ---
        p_config = p_config_base.copy()  # Work with a copy
        param_dim: int | None = None
        dataset_params = p_config.get("dataset_params", {})

        if "param_dim" in dataset_params:
            param_dim = int(dataset_params["param_dim"])
        elif "true_theta" in dataset_params:
            param_dim = len(dataset_params["true_theta"])
        else:
            if "synthetic" in p_name:
                raise ValueError(
                    f"Problem '{p_name}' is synthetic data, so param_dim or true_theta must be specified in the config."
                )
            print(f"   Inferring param_dim for problem '{p_name}' by loading the dataset...")
            dataset_name = p_config.get("dataset")
            if not dataset_name:
                raise ValueError(f"Problem config '{p_name}' is missing 'dataset' field.")

            loaded_data = load_dataset_from_source(dataset_name=dataset_name, random_state=0, **dataset_params)
            num_features = loaded_data["number_features"]
            bias = p_config.get("model_params", {}).get("bias", False)
            param_dim = num_features + 1 if bias else num_features
            print(f"   Inferred param_dim = {param_dim}")

        if param_dim is None:
            raise ValueError(f"Could not determine 'param_dim' for problem '{p_name}'.")

        if "dataset_params" not in p_config:
            raise ValueError(f"Problem config '{p_name}' is missing 'dataset_params' field.")
        p_config["dataset_params"]["param_dim"] = param_dim

        context = {"d": param_dim, "n": dataset_params.get("n_dataset", int(1e6))}

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

                if completion_manager.check_if_run_completed(unique_run_id):
                    print(f"--- Skipping already completed run (ID: {unique_run_id}) ---")
                    continue

                # --- W&B Init and Experiment Run ---
                wandb_run = None
                success = False
                try:
                    entity_to_use = WANDB_ENTITY
                    if entity_to_use:
                        print(f"--- Using entity from global WANDB_ENTITY variable: '{entity_to_use}' ---")
                    else:
                        try:
                            entity_to_use = wandb.api.default_entity
                            print(f"--- Using default wandb aentity: '{entity_to_use}' ---")
                        except AttributeError:
                            print(
                                "--- Could not determine default wandb entity. You may need to log in. Letting wandb decide. ---"
                            )
                            entity_to_use = None

                    sanitized_p_name = sanitize_for_wandb(p_name)
                    group_name = sanitized_o_name
                    wandb_run_name = unique_run_id

                    print(f"--- Starting run: {wandb_run_name} (Project: {sanitized_p_name}, Group: {group_name}) ---")

                    wandb_run = wandb.init(
                        entity=entity_to_use,
                        project=sanitized_p_name,
                        config=current_run_config,
                        group=group_name,
                        name=wandb_run_name,
                        mode="online",
                    )

                    actual_wandb_id = wandb_run.id
                    run_experiment(p_config, o_config, seed=i, project_name=p_name)

                    success = True
                    completion_manager.log_run_completion(unique_run_id, actual_wandb_id, p_name)
                    print(f"--- Finished and logged run (ID: {unique_run_id} / wandb: {actual_wandb_id}) ---")

                except Exception as e:
                    print(f"!!! ERROR during execution for {o_name}, seed {seed}: {e} !!!")
                    traceback.print_exc()
                    raise

                finally:
                    if wandb_run is not None:
                        exit_code = 0 if success else 1
                        wandb.finish(exit_code=exit_code)
                        print(f"--- WandB run finished for {o_name} (Exit code: {exit_code}) ---")


if __name__ == "__main__":
    main()
