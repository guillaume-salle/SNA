import os
import time
import wandb
import subprocess
import argparse
import yaml
import traceback
import hashlib
import socket

import torch
from torch.utils.data import DataLoader

from objective_functions import LinearRegression
from optimizers import SGD, USNA, BaseOptimizer
from datasets import generate_linear_regression


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
    dataset, true_theta, true_hessian = generate_linear_regression(**dataset_params, device=device)
    model = LinearRegression(**model_params)
    theta_init = true_theta + radius * torch.randn_like(true_theta)

    # --- Setup: Optimizer ---
    optimizer_class = get_optimizer_class(optimizer_name)
    optimizer = optimizer_class(param=theta_init, obj_function=model, **optimizer_params)

    # --- Training Loop ---
    dataloader = DataLoader(
        dataset,
        batch_size=optimizer.batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    n_samples_processed = 0
    start_time = time.time()
    initial_error = torch.linalg.vector_norm(optimizer.param - true_theta).item() ** 2
    wandb.log({"samples": n_samples_processed, "estimation_error": initial_error, "time": 0.0})
    print(f"   Starting optimization loop...")
    for step, data in enumerate(dataloader):
        if isinstance(data, (list, tuple)):
            X, y = data
            X, y = X.to(device), y.to(device)
            data = (X, y)
            current_batch_size = X.size(0)
        else:
            data = data.to(device)
            current_batch_size = data.size(0)
        optimizer.step(data)
        current_time = time.time()
        error = torch.linalg.vector_norm(optimizer.param - true_theta).item() ** 2
        n_samples_processed += current_batch_size
        log_data = {
            "samples": n_samples_processed,
            "estimation_error": error,
            "time": current_time - start_time,
        }
        wandb.log(log_data)

    # --- Final Logging ---
    final_time = time.time() - start_time
    print(f"   Finished optimization loop. Total time: {final_time:.2f}s")
    final_error = torch.linalg.vector_norm(optimizer.param - true_theta).item() ** 2
    print(f"   Final estimation error: {final_error:.4f}")
    wandb.log({"time": final_time, "estimation_error": final_error})


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


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run stochastic optimization experiments from config files.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the problem definition (dataset, model, radius).",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the optimizer (optimizer, params, device).",
    )
    parser.add_argument(
        "-N",
        "--N_runs",
        type=int,
        default=10,
        help="Number of runs (seeds) for averaging (default: 10, max: 100).",
    )

    args = parser.parse_args()

    # --- Load Configuration Files ---
    def load_config(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from: {config_path}")
            return config
        except FileNotFoundError:
            print(f"!!! ERROR: Configuration file not found at {config_path}. Exiting. !!!")
            exit(1)
        except yaml.YAMLError as e:
            print(f"!!! ERROR: Failed to parse configuration file {config_path}: {e}. Exiting. !!!")
            exit(1)
        except Exception as e:
            print(f"!!! ERROR: An unexpected error occurred loading config {config_path}: {e}. Exiting. !!!")
            exit(1)

    problem_config = load_config(args.config)
    optimizer_config = load_config(args.optimizer)

    # Check for device in optimizer config and set default if missing
    if "optimizer_params" not in optimizer_config:
        raise ValueError("optimizer_params not found in optimizer config")
    if "device" not in optimizer_config["optimizer_params"]:
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        optimizer_config["optimizer_params"]["device"] = default_device
        print(f"Device not specified in optimizer config, using default: {default_device}")

    # --- Validate N_runs ---
    requested_runs = args.N_runs
    max_runs = 100
    N_runs = max(1, min(requested_runs, max_runs))  # Clamp between 1 and max_runs
    if N_runs != requested_runs:
        print(f"Warning: Number of runs must be between 1 and {max_runs}. Using {N_runs} instead of {requested_runs}.")

    # --- Pre-Run Setup ---
    # Merge configs once for hashing and logging
    merged_config = problem_config.copy()
    merged_config.update(optimizer_config)
    base_identifier = hashlib.md5(config_to_stable_string(merged_config).encode()).hexdigest()

    # Calculate Project/Group names once
    problem_hash = hashlib.md5(config_to_stable_string(problem_config).encode()).hexdigest()[:8]
    project_name = f"{problem_config.get('dataset', 'unknown_dataset')}-{problem_hash}"
    optimizer_hash = hashlib.md5(config_to_stable_string(optimizer_config).encode()).hexdigest()[:8]
    group_name = f"{optimizer_config.get('optimizer', 'unknown_opt')}-{optimizer_hash}"

    print(f"Project Name: {project_name}")
    print(f"Group Name: {group_name}")
    print(f"Device to be used: {optimizer_config['optimizer_params']['device']}")
    print(f"Number of runs (seeds): {N_runs}")
    # Print dataset size (for generated dataset)
    if "dataset_params" in problem_config and "n_dataset" in problem_config["dataset_params"]:
        print(f"Dataset size: {problem_config['dataset_params']['n_dataset']}")
        try:
            # Convert n_dataset to int *within problem_config* before the loop
            problem_config["dataset_params"]["n_dataset"] = int(float(problem_config["dataset_params"]["n_dataset"]))
        except ValueError:
            print(f"!!! ERROR: Invalid value for 'n_dataset' in {args.config}. Exiting. !!!")
            exit(1)

    # --- Instantiate Completion Manager ---
    completion_manager = RunCompletionManager()  # Uses default log file path

    # --- Run Loop ---
    completed_runs_count = 0
    skipped_runs_count = 0
    current_run_config = merged_config.copy()

    for seed in range(N_runs):
        run_name = f"{optimizer_config.get('optimizer', 'unknown_opt')}-{base_identifier}_seed{seed}"
        print(f"\n--- Seed {seed}/{N_runs-1}: Checking run {run_name} ---")

        if completion_manager.check_if_run_completed(run_name):
            print(f"--- Skipping already completed run: {run_name} ---")
            skipped_runs_count += 1
            continue

        wandb_run = None  # Define wandb_run outside try block for finally
        success = False  # Flag to track if the run finished successfully
        current_run_config["seed"] = seed
        try:
            print(f"--- Starting run: {run_name} (Project: {project_name}, Group: {group_name}) ---")
            # Initialize WandB *before* calling run_experiment
            wandb_run = wandb.init(
                project=project_name,
                config=current_run_config,
                group=group_name,
                name=run_name,
                # mode="offline",
            )

            # Run the core experiment logic
            run_experiment(problem_config, optimizer_config, seed)

            # Mark successful completion *after* run_experiment finishes
            completion_manager.log_run_completion(run_name)
            completed_runs_count += 1
            success = True  # Mark as success before finishing
            print(f"--- Finished and logged run: {run_name} ---")

        except Exception as e:
            print(f"!!! ERROR during execution for seed {seed} (Run: {run_name}): {e} !!!")
            traceback.print_exc()
            raise e  # Stop script on first error

        finally:
            if wandb_run is not None:
                exit_code = 0 if success else 1
                wandb.finish(exit_code=exit_code)
                print(f"--- WandB run finished for {run_name} (Exit code: {exit_code}) ---")

    print(f"\n--- Overall Summary ---")
    print(f"Total seeds requested: {N_runs}")
    print(f"Runs skipped (already complete): {skipped_runs_count}")
    print(f"Runs completed successfully: {completed_runs_count}")

    # # --- Add Automatic Sync ---
    # def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    #     """
    #     Check for internet connection by trying to connect to a known host.
    #     Host: 8.8.8.8 (Google DNS)
    #     Port: 53/tcp (DNS)
    #     """
    #     try:
    #         socket.setdefaulttimeout(timeout)
    #         socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
    #         return True
    #     except socket.error:
    #         return False

    # print("\n--- Checking internet connection before attempting wandb sync... ---")
    # if check_internet_connection():
    #     print("--- Internet connection detected. Attempting to sync offline runs to wandb... ---")
    #     try:
    #         # Using check=True will raise an exception if the command fails
    #         print("wandb sync --sync-all")
    #         result = subprocess.run(["wandb", "sync", "--sync-all"], check=True, capture_output=True, text=True)
    #         print("--- wandb sync successful ---")
    #         print(result.stdout)  # Optionally print the output from wandb sync
    #         if result.stderr:
    #             print("--- wandb sync stderr: ---")
    #             print(result.stderr)
    #     except FileNotFoundError:
    #         print("!!! ERROR: 'wandb' command not found. Is wandb installed and in your PATH? Skipping sync. !!!")
    #     except subprocess.CalledProcessError as e:
    #         print(f"!!! ERROR: 'wandb sync --sync-all' failed with exit code {e.returncode} !!!")
    #         print("--- stdout ---")
    #         print(e.stdout)
    #         print("--- stderr ---")
    #         print(e.stderr)
    #         print("!!! Please check the errors above. You may need to run 'wandb sync --sync-all' manually. !!!")
    #     except Exception as e:
    #         print(f"!!! An unexpected error occurred during wandb sync: {e} !!!")
    #         print("!!! Skipping sync. You may need to run 'wandb sync --sync-all' manually. !!!")
    # else:
    #     print("--- No internet connection detected. Skipping wandb sync. ---")
    #     print("--- Please ensure you have an internet connection and run 'wandb sync --sync-all' manually later. ---")
    # # --- End Automatic Sync ---

    # print("\n--- Next Steps ---")
    # print("Go to the wandb project page:")
    # print("1. Use the 'Group' button/panel and group by 'wandb_group'.")
    # print("2. Select the group corresponding to your experiment.")
    # print("3. In the chart settings (pencil icon), enable 'Avg', 'StdDev', or 'Min/Max' aggregation.")
