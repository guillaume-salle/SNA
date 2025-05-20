import os
import time
import wandb
import subprocess
import argparse
import yaml
import traceback
import hashlib
import socket
import math
import sys
from tqdm import tqdm
import collections.abc  # For deep_merge

import torch
from torch.utils.data import DataLoader

from objective_functions import LinearRegression
from optimizers import SGD, USNA, BaseOptimizer
from datasets import generate_linear_regression


# ============================================================================ #
# >>> Configuration Utilities <<<                                              #
# ============================================================================ #


def deep_merge(source: dict, destination: dict) -> dict:
    """
    Deeply merges source dict into destination dict.
    Keys in source override keys in destination.
    Nested dictionaries are merged recursively.
    """
    for key, value in source.items():
        if isinstance(value, collections.abc.Mapping):
            # Get node or create one if it doesn't exist
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination


# ============================================================================ #
# >>> Run Completion Manager <<<                                               #
# ============================================================================ #


class RunCompletionManager:
    """
    Manages the completion log file and cache for tracking completed runs.
    """

    DEFAULT_LOG_FILE = ".completed_runs.log"

    def __init__(self, log_filepath: str = DEFAULT_LOG_FILE):
        """
        Initializes the manager with the path to the completion log file.

        Args:
            log_filepath (str): The path to the log file.
        """
        self.log_filepath = log_filepath
        self._completed_runs_cache: set[str] | None = None
        print(f"RunCompletionManager initialized with log file: {self.log_filepath}")

    def _read_log_file(self) -> None:
        """
        Reads the completion log file and populates the cache.
        """
        completed_runs = set()
        try:
            with open(self.log_filepath, "r") as f:
                for line in f:
                    completed_runs.add(line.strip())
            # Only print if file was actually read and had content potentially
            if completed_runs or os.path.exists(self.log_filepath):
                print(f"--> Read {len(completed_runs)} entries from completion log: {self.log_filepath}")
        except FileNotFoundError:
            print(f"--> Completion log file not found (normal for first run): {self.log_filepath}")
            # File doesn't exist yet, cache is an empty set
        except Exception as e:
            print(f"!!! Warning: Failed to read completion log file {self.log_filepath}: {e} !!!")
            # In case of error, don't trust potentially partial cache
            self._completed_runs_cache = None  # Invalidate cache on error
            raise  # Re-raise the exception after logging
        self._completed_runs_cache = completed_runs

    def check_if_run_completed(self, expected_run_name: str) -> bool:
        """
        Checks if a run identifier exists in the completion log file.

        Args:
            expected_run_name: The unique identifier (hash + seed) for the run.

        Returns:
            True if the run identifier is found in the log file, False otherwise.
        """
        if self._completed_runs_cache is None:
            self._read_log_file()
            # _read_log_file sets the cache, handle potential None if error occurred during read
            if self._completed_runs_cache is None:
                print("!!! Warning: Cache is None after attempting read, assuming run not completed due to read error.")
                return False  # Cannot confirm completion if read failed

        is_completed = expected_run_name in self._completed_runs_cache
        return is_completed

    def log_run_completion(self, run_name: str) -> None:
        """
        Logs a completed run name to the log file and updates the cache.

        Args:
            run_name (str): The unique identifier of the completed run.
        """
        try:
            # Append the unique run name to the log file
            with open(self.log_filepath, "a") as f:
                f.write(f"{run_name}\n")
            print(f"  [Completion Log] Added run to log: {self.log_filepath} -> {run_name}")

            # Update cache if it's already loaded
            if self._completed_runs_cache is not None:
                self._completed_runs_cache.add(run_name)
            # If cache wasn't loaded, it will be re-read on the next check
        except Exception as e:
            print(f"  [Completion Log] Warning: Failed to write to completion log {self.log_filepath}: {e}")
            # Invalidate cache if write fails, as its state might be inconsistent
            self._completed_runs_cache = None


# ============================================================================ #
# >>> Run Experiment <<<                                                       #
# ============================================================================ #


def get_optimizer_class(optimizer_name: str) -> BaseOptimizer:
    if optimizer_name == "SGD":
        return SGD
    elif optimizer_name == "USNA":
        return USNA
    else:
        raise ValueError(f"Unknown optimizer specified in config: {optimizer_name}")


def run_experiment(problem_config: dict, optimizer_config: dict, seed: int) -> None:
    """
    Runs the core optimization loop for a single experiment seed.
    Assumes wandb is already initialized.

    Args:
        problem_config (dict): The configuration dictionary for the problem.
        optimizer_config (dict): The configuration dictionary for the optimizer.
        seed (int): The seed for the experiment.
    """

    # --- Extract Parameters ---
    optimizer_params = optimizer_config.get("optimizer_params")
    device = optimizer_params.get("device")  # Should be present due to check in main
    radius = problem_config["radius"]
    optimizer_name = optimizer_config["optimizer"]
    model_params = problem_config.get("model_params")
    dataset_params = problem_config.get("dataset_params")

    # --- Setup: Data, Model, Initial Params ---
    torch.manual_seed(seed)
    data_gen_batch_size = optimizer_params["batch_size"]
    # Assuming generate_linear_regression returns: dataset, true_theta, true_hessian
    dataset, true_theta, true_hessian = generate_linear_regression(
        **dataset_params, device=device, data_batch_size=data_gen_batch_size
    )
    model = LinearRegression(**model_params)

    # Generate a random direction for the initial offset
    random_direction = torch.randn_like(true_theta)
    norm_of_random_direction = torch.linalg.vector_norm(random_direction)

    # Normalize the random direction to have unit norm
    # If norm_of_random_direction is 0 (e.g., if true_theta is an empty tensor for param_dim=0,
    # or in the extremely rare case randn_like produces a zero vector for non-empty true_theta),
    # the unit_direction will also be a zero vector (or empty tensor).
    # In such cases, theta_init will be equal to true_theta.
    if norm_of_random_direction == 0:
        unit_direction = random_direction  # This is already a zero vector or an empty tensor
    else:
        unit_direction = random_direction / norm_of_random_direction

    # Initialize theta_init on a sphere of 'radius' around true_theta
    theta_init = true_theta + radius * unit_direction

    # --- Setup: Optimizer ---
    optimizer_class = get_optimizer_class(optimizer_name)
    optimizer = optimizer_class(param=theta_init, obj_function=model, **optimizer_params)

    # --- Initial State Logging (before any optimizer steps) ---
    log_data_initial = {
        "samples": 1,  # Start samples at 1 for log scale compatibility
        "estimation_error": torch.linalg.vector_norm(theta_init - true_theta).item() ** 2,
        # Initial timing metrics for the first point
        "time": 0.0,
        "optimizer_time_cumulative": 0.0,
    }
    compute_inv_hess_error = True if true_hessian is not None and hasattr(optimizer, "matrix") else False
    if compute_inv_hess_error:
        true_inv_hessian = torch.linalg.inv(true_hessian)
        inv_hess_error = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord="fro").item()
        # compute also operator norm of the error
        inv_hess_error_operator = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord=2).item()
        log_data_initial["inv_hess_error"] = inv_hess_error
        log_data_initial["inv_hess_error_operator"] = inv_hess_error_operator
    wandb.log(log_data_initial)

    # --- Training Loop (No separate warm-up phase) ---
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    data_iterator = iter(dataloader)
    cumulative_optimizer_samples = 0  # Starts from 0, first step will process samples

    # total_batches_in_dataset is needed for tqdm if used
    total_batches_in_dataset = math.ceil(dataset.n_total_samples / dataset.data_batch_size)
    print(f"   Starting optimization loop for {total_batches_in_dataset} steps...")

    # Start timers for the main timed loop
    start_time = time.time()  # Overall wall-clock start time for the loop
    optimizer_time_cumulative = 0.0  # Cumulative time for optimizer.step() only

    progress_bar_iterator = tqdm(
        data_iterator, total=total_batches_in_dataset, desc="   Optimization", unit="batch", leave=True
    )

    for step, data_batch in enumerate(progress_bar_iterator):
        X, y = data_batch
        current_batch_size = X.size(0)

        current_step_duration: float
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            optimizer.step((X, y))
            end_event.record()
            torch.cuda.synchronize()
            current_step_duration = start_event.elapsed_time(end_event) / 1000.0
        else:  # CPU
            step_start_cpu_time = time.perf_counter()
            optimizer.step((X, y))
            step_end_cpu_time = time.perf_counter()
            current_step_duration = step_end_cpu_time - step_start_cpu_time

        optimizer_time_cumulative += current_step_duration
        cumulative_optimizer_samples += current_batch_size  # samples processed in this step

        loop_iteration_wall_time = time.time() - start_time
        error = torch.linalg.vector_norm(optimizer.param - true_theta).item() ** 2

        log_data_step = {
            "samples": 1 + cumulative_optimizer_samples,  # samples are 1 + (total processed by optimizer)
            "estimation_error": error,
            "time": loop_iteration_wall_time,
            "optimizer_time_cumulative": optimizer_time_cumulative,
        }
        if compute_inv_hess_error and hasattr(optimizer, "matrix"):
            inv_hess_error = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord="fro").item()
            inv_hess_error_operator = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord=2).item()
            log_data_step["inv_hess_error"] = inv_hess_error
            log_data_step["inv_hess_error_operator"] = inv_hess_error_operator
        if hasattr(optimizer, "log_metrics"):
            for key, value in optimizer.log_metrics.items():
                log_data_step[f"opt_{key}"] = value
        wandb.log(log_data_step)

    final_wall_time = time.time() - start_time
    print(f"\n   Finished optimization loop. Total wall time: {final_wall_time:.2f}s")
    print(f"   Total optimizer step time: {optimizer_time_cumulative:.4f}s")
    final_error = torch.linalg.vector_norm(optimizer.param - true_theta).item() ** 2
    print(f"   Final estimation error: {final_error:.4f}")

    # Final metrics
    average_optimizer_time_per_sample = optimizer_time_cumulative / cumulative_optimizer_samples
    average_optimizer_time_per_sample_ms = average_optimizer_time_per_sample * 1000.0
    log_data_final = {
        # "average_optimizer_time_per_sample": average_optimizer_time_per_sample,  # TODO : change to ms
        "average_optimizer_time_per_sample": average_optimizer_time_per_sample_ms,
    }

    # Log only the final optimizer metrics if they exist
    if hasattr(optimizer, "log_metrics_end") and isinstance(optimizer.log_metrics_end, dict):
        for key, value in optimizer.log_metrics_end.items():
            print(f"   Final optimizer metric: {key} = {value}")
            log_data_final[f"opt_{key}"] = value
    else:
        print("   No final optimizer metrics to log")

    wandb.log(log_data_final)


# ============================================================================ #
# >>> Main                                                                     #
# ============================================================================ #


# Function to recursively convert dict to a sorted, stable string
def config_to_stable_string(cfg_item):
    if isinstance(cfg_item, dict):
        # Sort keys, recursively process values
        return "{" + ",".join(f"{k}:{config_to_stable_string(v)}" for k, v in sorted(cfg_item.items())) + "}"
    elif isinstance(cfg_item, list):
        # Process list items recursively
        return "[" + ",".join(config_to_stable_string(i) for i in cfg_item) + "]"
    elif isinstance(cfg_item, tuple):
        # Process tuple items recursively
        return "(" + ",".join(config_to_stable_string(i) for i in cfg_item) + ")"
    else:
        # Convert other types to string
        return str(cfg_item)


# Basic sanitization function
def sanitize_name(name):
    return (
        name.replace("{", "")
        .replace("}", "")
        .replace(":", "-")
        .replace(",", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace(" ", "")
        .replace("=", "-")
    )


# Define the subdirectory where optimizer YAML configurations are stored
OPTIMIZER_CONFIGS_DIR = "configs/optimizers/"
# Define the subdirectory where problem YAML configurations are stored
PROBLEM_CONFIGS_DIR = "configs/problem/"

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run stochastic optimization experiments from config files.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML problem configuration file.",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        nargs="+",
        required=True,
        help="Paths or names of YAML optimizer configuration files (e.g., SGD or SGD.yaml for configs/optimizers/SGD.yaml, or a direct path like /path/to/my_opt.yaml). Shell globbing (e.g., 'configs/optimizers/*') is also supported.",
    )
    parser.add_argument(
        "-N",
        "--N_runs",
        type=int,
        default=1,
        help="Number of runs (seeds) for each optimizer config",
    )

    args = parser.parse_args()

    problem_config_path_arg = args.config

    # --- Load Configuration Files ---
    def load_config(config_path, _load_chain=None):
        if _load_chain is None:
            _load_chain = set()
        # Use os.path.normpath and os.path.abspath to get a canonical path for _load_chain
        # to prevent issues with slightly different relative path strings referring to the same file.
        normalized_config_path = os.path.normpath(os.path.abspath(config_path))
        if normalized_config_path in _load_chain:
            raise ValueError(
                f"Circular inheritance detected: {normalized_config_path} already in load chain: {_load_chain}"
            )

        _load_chain.add(normalized_config_path)

        try:
            with open(config_path, "r") as f:
                current_config = yaml.safe_load(f)
            if not isinstance(current_config, dict):
                print(f"Warning: Config file {config_path} is empty or not a dictionary. Returning empty config.")
                current_config = {}

            print(f"Loaded initial configuration from: {config_path}")

            parent_config_filename = current_config.pop("inherits_from", None)

            if parent_config_filename:
                actual_parent_filename = parent_config_filename
                if not (actual_parent_filename.endswith(".yaml") or actual_parent_filename.endswith(".yml")):
                    actual_parent_filename += ".yaml"

                # Resolve parent path relative to the directory of the current config file
                current_config_dir = os.path.dirname(config_path)
                parent_config_path = os.path.join(current_config_dir, actual_parent_filename)

                print(f"  File {config_path} inherits from {parent_config_filename} (resolved to {parent_config_path})")
                base_config = load_config(parent_config_path, _load_chain=_load_chain.copy())

                final_config = deep_merge(current_config, base_config)
                _load_chain.remove(normalized_config_path)
                return final_config
            else:
                _load_chain.remove(normalized_config_path)
                return current_config

        except FileNotFoundError:
            print(f"!!! ERROR: Configuration file not found at {config_path}. Exiting. !!!")
            _load_chain.remove(normalized_config_path)
            exit(1)
        except yaml.YAMLError as e:
            print(f"!!! ERROR: Failed to parse configuration file {config_path}: {e}. Exiting. !!!")
            _load_chain.remove(normalized_config_path)
            exit(1)
        except Exception as e:
            print(f"!!! ERROR: An unexpected error occurred loading config {config_path}: {e}. Exiting. !!!")
            # Ensure removal even if normalized_config_path itself was problematic, though less likely here.
            if normalized_config_path in _load_chain:
                _load_chain.remove(normalized_config_path)
            exit(1)

    # --- Resolve and load the main problem configuration ---
    problem_config_argument = args.config
    problem_config_filename_base_for_logging = os.path.splitext(os.path.basename(problem_config_argument))[0]

    candidate_problem_path = problem_config_argument
    if not (candidate_problem_path.endswith(".yaml") or candidate_problem_path.endswith(".yml")):
        candidate_problem_path += ".yaml"

    if os.path.isfile(candidate_problem_path):
        actual_problem_config_filepath = candidate_problem_path
    else:
        filename_component = os.path.basename(problem_config_argument)  # Use basename of original arg for dir lookup
        if not (filename_component.endswith(".yaml") or filename_component.endswith(".yml")):
            filename_component += ".yaml"
        actual_problem_config_filepath = os.path.join(PROBLEM_CONFIGS_DIR, filename_component)

    print(
        f"Attempting to load problem configuration: '{problem_config_argument}' (resolved to: '{actual_problem_config_filepath}')"
    )
    problem_config = load_config(actual_problem_config_filepath)

    # --- Process optimizer arguments ---
    optimizer_config_file_args = []
    for optimizer_arg in args.optimizer:
        # Handle glob patterns
        if "*" in optimizer_arg or "?" in optimizer_arg:
            import glob

            expanded_paths = glob.glob(optimizer_arg)
            if not expanded_paths:
                print(f"Warning: No files found matching pattern: {optimizer_arg}")
                continue
            optimizer_config_file_args.extend(expanded_paths)
        else:
            optimizer_config_file_args.append(optimizer_arg)

    if not optimizer_config_file_args:
        print("Error: No valid optimizer configurations found. Exiting.")
        exit(1)

    # --- Hash and project name are derived from the loaded problem_config and its resolved path ---
    # Note: project_name uses the original argument's basename for user-friendliness if desired,
    # or could use the basename of actual_problem_config_filepath.
    # For consistency in hashing the content, actual_problem_config_filepath should be what matters for content hash.
    # The problem_config_filename_base_for_logging is for the human-readable part of the project name.

    problem_content_stable_string = config_to_stable_string(problem_config)  # Hash the actual loaded content
    problem_hash = hashlib.md5(problem_content_stable_string.encode()).hexdigest()[:8]
    # Use problem_config_filename_base_for_logging for the friendly name part
    project_name = f"{problem_config_filename_base_for_logging}-{problem_hash}"

    # --- Validate N_runs ---
    requested_runs = args.N_runs
    max_runs = 100
    N_runs = max(1, min(requested_runs, max_runs))  # Clamp between 1 and max_runs
    if N_runs != requested_runs:
        print(f"Warning: Number of runs must be between 1 and {max_runs}. Using {N_runs} instead of {requested_runs}.")

    # --- Instantiate Completion Manager (once) ---
    completion_manager = RunCompletionManager()

    # --- Determine param_dim for batch_size defaulting (if needed) ---
    param_dim_for_config: int
    dataset_params = problem_config.get("dataset_params", {})
    if dataset_params.get("true_theta") is not None:
        param_dim_for_config = len(dataset_params["true_theta"])
    elif dataset_params.get("param_dim") is not None:
        param_dim_for_config = int(dataset_params["param_dim"])
    else:
        raise ValueError(
            "Cannot determine param_dim from problem_config for batch_size setup. "
            "Please provide 'true_theta' or 'param_dim' in dataset_params."
        )

    if "dataset_params" in problem_config and "n_dataset" in problem_config["dataset_params"]:
        try:
            current_n_dataset = int(float(problem_config["dataset_params"]["n_dataset"]))
            problem_config["dataset_params"]["n_dataset"] = current_n_dataset  # Ensure it's updated in dict passed
            print(f"Dataset size: {current_n_dataset}")
        except ValueError:
            print(f"!!! ERROR: Invalid value for 'n_dataset' in problem_config {args.config}. Exiting. !!!")
            exit(1)

    # --- Loop over each optimizer configuration argument ---
    for optimizer_file_argument in optimizer_config_file_args:

        optimizer_config_filename_base = os.path.splitext(os.path.basename(optimizer_file_argument))[0]

        candidate_optimizer_path = optimizer_file_argument
        if not (candidate_optimizer_path.endswith(".yaml") or candidate_optimizer_path.endswith(".yml")):
            candidate_optimizer_path += ".yaml"

        if os.path.isfile(candidate_optimizer_path):
            actual_optimizer_config_filepath = candidate_optimizer_path
        else:
            filename_component = os.path.basename(optimizer_file_argument)
            if not (filename_component.endswith(".yaml") or filename_component.endswith(".yml")):
                filename_component += ".yaml"
            actual_optimizer_config_filepath = os.path.join(OPTIMIZER_CONFIGS_DIR, filename_component)

        print(f"\n============================================================================")
        print(
            f"Processing Optimizer Argument: '{optimizer_file_argument}' (using base name: '{optimizer_config_filename_base}', resolved to load: '{actual_optimizer_config_filepath}')"
        )
        print(f"============================================================================\n")
        # Use the same load_config for optimizers; it now handles relative inheritance correctly
        optimizer_config = load_config(actual_optimizer_config_filepath)

        # --- Per-Optimizer Config Setup ---
        if "optimizer_params" not in optimizer_config:
            raise ValueError(f"optimizer_params not found in optimizer config: {actual_optimizer_config_filepath}")
        optimizer_params = optimizer_config["optimizer_params"]

        if "device" not in optimizer_params:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            optimizer_params["device"] = default_device
            print(f"Device not specified in {actual_optimizer_config_filepath}, using default: {default_device}")

        # --- Set or default batch_size in optimizer_config["optimizer_params"] and ensure it's an int ---
        if optimizer_params.get("batch_size") is None:
            batch_size_power = optimizer_params.get("batch_size_power", BaseOptimizer.DEFAULT_BATCH_SIZE_POWER)
            calculated_batch_size = int(param_dim_for_config**batch_size_power)
            optimizer_params["batch_size"] = calculated_batch_size
            print(f"Calculated batch_size: {calculated_batch_size} for param dim: {param_dim_for_config}")
        else:
            try:
                optimizer_params["batch_size"] = int(float(optimizer_params["batch_size"]))
            except ValueError:
                raise ValueError(
                    f"Invalid batch_size in {actual_optimizer_config_filepath}: {optimizer_params['batch_size']}. Must be a number convertible to int."
                )

        # --- Pre-Run Setup (per optimizer config) ---
        optimizer_hash = hashlib.md5(config_to_stable_string(optimizer_config).encode()).hexdigest()[:6]
        group_name = f"{optimizer_config_filename_base}-{optimizer_hash}"

        merged_config = problem_config.copy()
        merged_config.update(optimizer_config)
        run_identifier = hashlib.md5(config_to_stable_string(merged_config).encode()).hexdigest()[:6]

        print(f"--- Configuration for {optimizer_config_filename_base} ---")
        print(f"Project Name: {project_name}")
        print(f"Group Name: {group_name}")
        print(f"Device to be used: {optimizer_params['device']}")
        print(f"Number of runs (seeds): {N_runs}")

        # --- Run Loop (per optimizer config, over seeds) ---
        completed_runs_count = 0
        skipped_runs_count = 0

        for seed in range(N_runs):
            current_run_config = merged_config.copy()
            current_run_config["seed"] = seed
            run_name = f"{optimizer_config_filename_base}-{run_identifier}_{seed}"

            print(
                f"\n--- Seed {seed}/{N_runs-1}: Checking run {run_name} for optimizer {optimizer_config_filename_base} ---"
            )

            if completion_manager.check_if_run_completed(run_name):
                print(f"--- Skipping already completed run: {run_name} ---")
                skipped_runs_count += 1
                continue

            wandb_run = None
            success = False
            try:
                print(f"--- Starting run: {run_name} (Project: {project_name}, Group: {group_name}) ---")
                wandb_run = wandb.init(
                    entity="USNA",
                    project=project_name,
                    config=current_run_config,  # Pass the config specific to this run (optimizer + seed)
                    group=group_name,
                    name=run_name,
                    mode="online",
                )

                run_experiment(problem_config, optimizer_config, seed)  # Pass the current optimizer_config

                completion_manager.log_run_completion(run_name)
                completed_runs_count += 1
                success = True
                print(f"--- Finished and logged run: {run_name} ---")

            except Exception as e:
                print(
                    f"!!! ERROR during execution for {optimizer_config_filename_base}, seed {seed} (Run: {run_name}): {e} !!!"
                )
                traceback.print_exc()
                # Decide if one error should stop all processing or just for this optimizer_config
                # For now, let's re-raise to stop everything, consistent with previous behavior.
                raise e

            finally:
                if wandb_run is not None:
                    exit_code = 0 if success else 1
                    wandb.finish(exit_code=exit_code)
                    print(f"--- WandB run finished for {run_name} (Exit code: {exit_code}) ---")

        print(f"\n--- Summary for Optimizer {optimizer_config_filename_base} ---")
        print(f"Total seeds requested: {N_runs}")
        print(f"Runs skipped (already complete): {skipped_runs_count}")
        print(f"Runs completed successfully: {completed_runs_count}")
        print(f"----------------------------------------------------------------------------")

    # --- Overall Summary  ---
    print(f"\n--- All optimizer configurations processed. ---")
