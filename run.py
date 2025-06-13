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

from objective_functions import LinearRegression, LogisticRegression
from optimizers import SGD, mSNA, BaseOptimizer
from datasets import generate_regression, load_dataset_from_source


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


def evaluate_expression(expr: str, context: dict) -> float:
    """
    Safely evaluate a mathematical expression with variables from context.
    The context includes global variables like 'd' and local variables
    from the same config block.
    """
    # Create a safe environment with only allowed operations
    safe_dict = {
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "round": round,
        "sum": sum,
        "int": int,
        "float": float,
        "math": math,
    }
    # The context contains both global variables (like 'd') and
    # already-evaluated local variables from the config block.
    safe_dict.update(context)

    try:
        # Eval is used here, but with a carefully controlled and safe scope.
        return eval(expr, {"__builtins__": {}}, safe_dict)
    except NameError:
        # Re-raise NameError specifically. This allows the calling function
        # to catch it and handle dependency-based retries.
        raise
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {str(e)}")


def process_config_values(config: dict, context: dict) -> dict:
    """
    Process config values, evaluating expressions marked with 'expr:'.
    This function handles dependencies between expressions in the same block.
    """
    processed_config = {}
    unprocessed_expressions = {}

    # First pass: process non-expressions and identify all expressions
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            # Pass a copy of the context to avoid child contexts polluting parent contexts
            processed_config[key] = process_config_values(value, context.copy())
        elif isinstance(value, str) and value.startswith("expr:"):
            unprocessed_expressions[key] = value[5:].strip()
        else:
            processed_config[key] = value

    # Create a local context for expression evaluation within this block
    # It starts with a copy of the global context
    local_context = context.copy()
    # And is updated with the non-expression values we just processed
    local_context.update(processed_config)

    # Second pass: iteratively evaluate expressions, handling dependencies
    while unprocessed_expressions:
        processed_this_round = []
        for key, expr in unprocessed_expressions.items():
            try:
                # Attempt to evaluate the expression with the current local_context
                eval_result = evaluate_expression(expr, local_context)
                processed_config[key] = eval_result
                local_context[key] = eval_result  # Add newly evaluated value to the context
                processed_this_round.append(key)
            except NameError:
                # This expression depends on another one not yet processed.
                # We will retry in the next iteration.
                continue
            except Exception as e:
                # Handle other potential evaluation errors
                raise ValueError(f"Error processing expression for key '{key}': {e}")

        if not processed_this_round:
            # If we went through a whole round without processing anything,
            # it means there's a circular dependency.
            raise ValueError(
                f"Circular dependency or undefined variable in expressions: {list(unprocessed_expressions.keys())}"
            )

        # Remove the processed keys for the next iteration
        for key in processed_this_round:
            del unprocessed_expressions[key]

    return processed_config


# ============================================================================ #
# >>> Run Completion Manager <<<                                               #
# ============================================================================ #


class RunCompletionManager:
    """
    Manages the completion log file and cache for tracking completed runs.
    The log file is structured with project names as headers.
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
        The file is expected to be structured with project names as headers.
        """
        completed_runs = set()
        try:
            with open(self.log_filepath, "r") as f:
                for line in f:
                    # A run is an indented line. This correctly ignores project headers.
                    if line.strip() and not line.endswith(":") and (line.startswith(" ") or line.startswith("\t")):
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

    def log_run_completion(self, run_name: str, project_name: str) -> None:
        """
        Logs a completed run name under its project header in the log file.
        This method is not thread-safe but is sufficient for sequential runs.

        Args:
            run_name (str): The unique identifier of the completed run.
            project_name (str): The name of the project for grouping.
        """
        try:
            # Read all existing lines from the log file
            try:
                with open(self.log_filepath, "r") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                lines = []

            # Prepare the new entries
            project_header = f"{project_name}:"
            new_run_line = f"    {run_name}\n"

            project_index = -1
            for i, line in enumerate(lines):
                if line.strip() == project_header:
                    project_index = i
                    break

            if project_index != -1:
                # Find where to insert the new run within the project block
                insert_index = project_index + 1
                while insert_index < len(lines) and (
                    lines[insert_index].startswith(" ") or lines[insert_index].startswith("\t")
                ):
                    insert_index += 1
                lines.insert(insert_index, new_run_line)
            else:  # Project header not found, so add it
                # To keep it clean, add a newline before a new project if the file is not empty
                if lines and not lines[-1].endswith("\n"):
                    lines.append("\n")
                lines.append(f"{project_header}\n")
                lines.append(new_run_line)

            # Write the updated content back to the file
            with open(self.log_filepath, "w") as f:
                f.writelines(lines)

            print(f"  [Completion Log] Added run to log under project '{project_name}': {run_name}")

            # Update cache if it's already loaded
            if self._completed_runs_cache is not None:
                self._completed_runs_cache.add(run_name)
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
        return mSNA
    else:
        raise ValueError(f"Unknown optimizer specified in config: {optimizer_name}")


def run_experiment(problem_config: dict, optimizer_config: dict, seed: int, project_name: str) -> None:
    """
    Runs the core optimization loop for a single experiment seed.
    Assumes wandb is already initialized.

    Args:
        problem_config (dict): The configuration dictionary for the problem.
        optimizer_config (dict): The configuration dictionary for the optimizer.
        seed (int): The seed for the experiment.
        project_name (str): The name of the project for logging completion.
    """

    # --- Extract Parameters ---
    optimizer_params = optimizer_config.get("optimizer_params")
    device = optimizer_params.get("device")  # Should be present due to check in main
    radius = problem_config.get("radius")
    optimizer_name = optimizer_config.get("optimizer")
    dataset_name = problem_config.get("dataset")
    dataset_params = problem_config.get("dataset_params")
    model_name = problem_config.get("model")
    model_params = problem_config.get("model_params")

    # --- Setup: Model ---
    # Instantiate the model early, as it's needed for Hessian estimation.
    if model_name == "LinearRegression":
        model = LinearRegression(**model_params)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unknown model type specified in problem_config: {model_name}")

    # --- Setup: Data, Model, Initial Params ---
    # Ensure global seed is set for other one-off random operations (e.g., theta_init)
    torch.manual_seed(seed)
    batch_size = optimizer_params["batch_size"]  # This is the data_batch_size for the iterable dataset

    param_dim: int
    train_set: torch.utils.data.Dataset
    test_set: torch.utils.data.Dataset | None = None

    if dataset_name in ["synthetic_linear_regression", "synthetic_logistic_regression"]:
        # --- Main Training Dataset ---
        # Define a device for the data generation process itself.
        data_gen_device = device if "cuda" in str(device) else "cpu"

        # Create and seed a dedicated RNG for the main training data.
        rng_train = torch.Generator(device=data_gen_device)
        rng_train.manual_seed(seed)

        # Generate the main training set
        train_set, true_theta, true_hessian = generate_regression(
            dataset_name=dataset_name,
            dataset_params=dataset_params,
            device=device,
            data_batch_size=batch_size,
            rng_data=rng_train,
            data_gen_device=data_gen_device,
        )
        param_dim = true_theta.shape[0]
        n_train_set = dataset_params.get("n_dataset")

        if true_hessian is None and true_theta is not None:
            print(
                f"   Estimating true_hessian for {dataset_name} at true_theta as it was not returned by generate_regression..."
            )

            # --- Hessian Estimation Dataset ---
            # Use a large batch size for efficiency
            hessian_estimation_batch_size = 2048
            # Cap the number of samples for Hessian estimation for efficiency and precision
            n_samples_for_hessian_estimation = int(1e7)

            print(
                f"   Using a temporary dataset of {n_samples_for_hessian_estimation} samples (batch size: {hessian_estimation_batch_size}) for this estimation."
            )

            # Create a temporary config for the Hessian dataset with the capped sample size
            dataset_params_hessian = dataset_params.copy()
            dataset_params_hessian["n_dataset"] = n_samples_for_hessian_estimation

            # Create and seed a separate, identically-seeded RNG for Hessian estimation
            # to ensure the data sequence is identical but the generator state is isolated.
            rng_hessian = torch.Generator(device=data_gen_device)
            rng_hessian.manual_seed(seed)

            hessian_dataset, _, _ = generate_regression(
                dataset_name=dataset_name,
                dataset_params=dataset_params_hessian,  # Use the modified params
                device=device,
                data_batch_size=hessian_estimation_batch_size,
                rng_data=rng_hessian,  # Use the separate generator
                data_gen_device=data_gen_device,
            )

            accumulated_hessian = torch.zeros((param_dim, param_dim), device=device, dtype=true_theta.dtype)

            hessian_estimation_loader = DataLoader(
                hessian_dataset,  # Use the temporary dataset
                batch_size=None,  # Pass batches from dataset as-is
                shuffle=False,
                pin_memory=(device == "cuda"),
            )

            for X_batch_hess_cpu, y_batch_hess_cpu in hessian_estimation_loader:
                # Move data to the model device for the model's Hessian calculation
                X_batch_hess, y_batch_hess = X_batch_hess_cpu.to(device), y_batch_hess_cpu.to(device)

                batch_hess = model.hessian((X_batch_hess, y_batch_hess), true_theta)
                accumulated_hessian += batch_hess * X_batch_hess.shape[0]

            true_hessian = accumulated_hessian / n_samples_for_hessian_estimation
            print(f"   Finished estimating true_hessian. Averaged over {n_samples_for_hessian_estimation:.0e} samples.")

    else:
        print(f"   Loading real dataset: {dataset_name}...")
        train_set, test_set, param_dim, n_train_set, n_test_set = load_dataset_from_source(
            dataset_name=dataset_name, device=device, **dataset_params
        )
        true_theta, true_hessian = None, None

    # Initialize theta_init on a sphere of 'radius' around true_theta
    if true_theta is not None:
        # For synthetic data, initialize around true_theta using radius
        random_direction = torch.randn_like(true_theta)
        random_direction /= torch.linalg.vector_norm(random_direction)
        theta_init = true_theta + radius * random_direction
    else:
        print(f"Initializing theta_init to zeros for real dataset (param_dim: {param_dim})")
        theta_init = torch.zeros(param_dim, device=device, dtype=torch.float32)

    # --- Setup: Optimizer ---
    optimizer_class = get_optimizer_class(optimizer_name)
    optimizer = optimizer_class(param=theta_init, obj_function=model, **optimizer_params)

    # --- Initial State Logging (before any optimizer steps) ---
    log_data_initial = {
        "samples": 1,  # Start samples at 1 for log scale compatibility
        "time": 0.0,
        "optimizer_time_cumulative": 0.0,
    }
    compute_theta_error = True if true_theta is not None else False
    if compute_theta_error:
        log_data_initial["estimation_error"] = torch.linalg.vector_norm(theta_init - true_theta).item() ** 2
    compute_inv_hess_error = True if true_hessian is not None and hasattr(optimizer, "matrix") else False
    compute_inv_hess_error_avg = (
        True if true_hessian is not None and getattr(optimizer, "averaged_matrix", False) else False
    )
    if compute_inv_hess_error:
        true_inv_hessian = torch.linalg.inv(true_hessian)
        del true_hessian  # free memory

        # Calculate and print eigenvalues of the true inverse Hessian
        inv_hess_eigenvalues = torch.linalg.eigvalsh(true_inv_hessian)
        lambda_min_inv_hess = inv_hess_eigenvalues.min().item()
        lambda_max_inv_hess = inv_hess_eigenvalues.max().item()
        print(f"   True Inv Hessian: lambda_min={lambda_min_inv_hess:.4f}, lambda_max={lambda_max_inv_hess:.4f}")

        inv_hess_error_fro = torch.linalg.norm(true_inv_hessian - optimizer.matrix_not_avg, ord="fro").item() ** 2
        inv_hess_error_operator = torch.linalg.norm(true_inv_hessian - optimizer.matrix_not_avg, ord=2).item()
        log_data_initial["inv_hess_error_fro"] = inv_hess_error_fro
        log_data_initial["inv_hess_error_operator"] = inv_hess_error_operator
        if compute_inv_hess_error_avg:
            inv_hess_error_fro_avg = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord="fro").item() ** 2
            inv_hess_error_operator_avg = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord=2).item()
            log_data_initial["inv_hess_error_fro_avg"] = inv_hess_error_fro_avg
            log_data_initial["inv_hess_error_operator_avg"] = inv_hess_error_operator_avg
    wandb.log(log_data_initial)

    # --- Training Loop (No separate warm-up phase) ---
    # The dataloader will use the `train_set` which is already configured
    # with its own independent, seeded generator (`rng_train`). No further seeding is needed here.
    dataloader = DataLoader(
        train_set,
        batch_size=None,  # Correctly None for IterableDataset yielding batches
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    data_iterator = iter(dataloader)
    cumulative_optimizer_samples = 0  # Starts from 0, first step will process samples

    # total_batches_in_dataset is needed for tqdm
    total_samples_for_tqdm = n_train_set
    total_batches_in_dataset = math.ceil(total_samples_for_tqdm / batch_size)

    print(
        f"   Starting optimization loop for {total_batches_in_dataset if total_batches_in_dataset is not None else 'unknown'} steps..."
    )

    # Start timers for the main timed loop
    start_time = time.time()  # Overall wall-clock start time for the loop
    optimizer_time_cumulative = 0.0  # Cumulative time for optimizer.step() only

    progress_bar_iterator = tqdm(
        data_iterator, total=total_batches_in_dataset, desc="   Optimization", unit="batch", leave=True
    )

    for _, data_batch in enumerate(progress_bar_iterator):
        # data_batch is a tuple of CPU tensors yielded from our dataset
        X_cpu, y_cpu = data_batch
        current_batch_size = X_cpu.size(0)

        current_step_duration: float
        if device == "cuda":
            # Time the data transfer and the optimizer step together for GPU
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            X = X_cpu.to(device)
            y = y_cpu.to(device)
            optimizer.step((X, y))
            end_event.record()

            torch.cuda.synchronize()
            current_step_duration = start_event.elapsed_time(end_event) / 1000.0
        else:  # CPU
            # For CPU, the .to(device) is a no-op but we include it for structural consistency.
            # The timing correctly captures just the optimizer step.
            step_start_cpu_time = time.perf_counter()
            X = X_cpu.to(device)
            y = y_cpu.to(device)
            optimizer.step((X, y))
            step_end_cpu_time = time.perf_counter()
            current_step_duration = step_end_cpu_time - step_start_cpu_time

        optimizer_time_cumulative += current_step_duration
        cumulative_optimizer_samples += current_batch_size  # samples processed in this step

        loop_iteration_wall_time = time.time() - start_time

        log_data_step = {
            "samples": 1 + cumulative_optimizer_samples,  # samples are 1 + (total processed by optimizer)
            "time": loop_iteration_wall_time,
            "optimizer_time_cumulative": optimizer_time_cumulative,
        }
        if compute_theta_error:
            log_data_step["estimation_error"] = torch.linalg.vector_norm(optimizer.param - true_theta).item() ** 2

        if compute_inv_hess_error:
            inv_hess_error_fro = torch.linalg.norm(true_inv_hessian - optimizer.matrix_not_avg, ord="fro").item() ** 2
            inv_hess_error_operator = torch.linalg.norm(true_inv_hessian - optimizer.matrix_not_avg, ord=2).item()
            log_data_step["inv_hess_error_fro"] = inv_hess_error_fro
            log_data_step["inv_hess_error_operator"] = inv_hess_error_operator

            if compute_inv_hess_error_avg:
                inv_hess_error_fro_avg = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord="fro").item() ** 2
                inv_hess_error_operator_avg = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord=2).item()
                log_data_step["inv_hess_error_fro_avg"] = inv_hess_error_fro_avg
                log_data_step["inv_hess_error_operator_avg"] = inv_hess_error_operator_avg

        if hasattr(optimizer, "log_metrics"):
            for key, value in optimizer.log_metrics.items():
                log_data_step[f"opt_{key}"] = value
        wandb.log(log_data_step)

    final_wall_time = time.time() - start_time
    print(f"\n   Finished optimization loop. Total wall time: {final_wall_time:.2f}s")
    print(f"   Total optimizer step time: {optimizer_time_cumulative:.4f}s")

    # Final metrics
    log_data_final = {}

    if compute_theta_error:
        final_error = torch.linalg.vector_norm(optimizer.param - true_theta).item() ** 2
        print(f"   Final estimation error: {final_error:.4f}")
    else:
        # Compute and log test accuracy if test_set is available
        if test_set is not None:
            print(f"   Calculating accuracy and loss on the test set...")
            test_loader = DataLoader(test_set, batch_size=256)  # Use a larger batch for eval

            all_predictions = []
            all_targets = []
            total_loss = 0.0
            with torch.no_grad():
                for X_batch_cpu, y_batch_cpu in test_loader:
                    X_batch, y_batch = X_batch_cpu.to(device), y_batch_cpu.to(device)

                    # --- Loss Calculation ---
                    # The model is the objective function, call it directly.
                    # It returns loss averaged over the batch, so we multiply by batch size to get total.
                    loss = model((X_batch, y_batch), optimizer.param)
                    total_loss += loss.item() * X_batch.size(0)

                    # --- Accuracy Calculation (currently only for Logistic Regression) ---
                    if model_name == "LogisticRegression":
                        phi_batch = model._add_bias(X_batch)
                        logits = torch.matmul(phi_batch, optimizer.param)
                        predictions = (torch.sigmoid(logits) > 0.5).float()
                        all_predictions.append(predictions.cpu())
                        all_targets.append(y_batch_cpu.squeeze())
                    else:
                        # For other models, we can still calculate loss, but accuracy logic is specific.
                        # We just pass here and don't populate accuracy metrics.
                        pass

            # --- Log Final Metrics ---
            total_samples = len(test_set)
            if total_samples > 0:
                # Log average test loss
                average_loss = total_loss / total_samples
                print(f"   Test Set Avg Loss: {average_loss:.4f}")
                log_data_final["test_set_loss"] = average_loss

            if all_predictions and all_targets:
                # Log test accuracy
                predictions_tensor = torch.cat(all_predictions)
                targets_tensor = torch.cat(all_targets)
                correct_predictions = (predictions_tensor == targets_tensor).sum().item()
                accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

                print(f"   Test Set Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
                log_data_final["test_set_accuracy"] = accuracy
            else:
                print(f"   Skipped test set accuracy calculation (not applicable for this model type).")

    # Final metrics
    log_data_final["optimizer_time"] = optimizer_time_cumulative

    # Log the final optimizer metrics, if they exist
    if hasattr(optimizer, "log_metrics_end") and isinstance(optimizer.log_metrics_end, dict):
        for key, value in optimizer.log_metrics_end.items():
            print(f"   Final optimizer metric: {key} = {value}")
            log_data_final[f"opt_{key}"] = value
    else:
        print("   No final optimizer metrics to log")

    print(log_data_final)
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


# Define the subdirectory where problem YAML configurations are stored
PROBLEM_CONFIGS_DIR = "configs/problems/"
# Define the subdirectory for base optimizer configurations used for inheritance
OPTIMIZER_BASE_DIR = "configs/optimizers_base/"

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

                # First try to resolve parent path relative to the directory of the current config file
                current_config_dir = os.path.dirname(config_path)
                parent_config_path = os.path.join(current_config_dir, actual_parent_filename)

                # If parent not found in same directory, try in OPTIMIZER_BASE_DIR
                if not os.path.isfile(parent_config_path):
                    parent_config_path = os.path.join(OPTIMIZER_BASE_DIR, actual_parent_filename)

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

    # Create context from problem config for expression evaluation
    context = {}
    if "dataset_params" in problem_config:
        if "true_theta" in problem_config["dataset_params"]:
            # If true_theta is provided, use its length for d
            true_theta = problem_config["dataset_params"]["true_theta"]
            if isinstance(true_theta, list):
                context["d"] = len(true_theta)
            else:
                raise ValueError("true_theta must be a list in the config file")
        elif "param_dim" in problem_config["dataset_params"]:
            context["d"] = problem_config["dataset_params"]["param_dim"]
        if "n_dataset" in problem_config["dataset_params"]:
            context["n"] = problem_config["dataset_params"]["n_dataset"]

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

    # --- Hash and project name are derived from the loaded problem_config ---
    problem_content_stable_string = config_to_stable_string(problem_config)  # Hash the actual loaded content
    problem_hash = hashlib.md5(problem_content_stable_string.encode()).hexdigest()[:8]
    # Use problem_config_filename_base_for_logging for the friendly name part
    project_name = f"{problem_config_filename_base_for_logging}"

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
        actual_optimizer_config_filepath = optimizer_file_argument

        if not os.path.isfile(actual_optimizer_config_filepath):
            print(
                f"!!! ERROR: Optimizer configuration file not found: '{actual_optimizer_config_filepath}'. Skipping. !!!"
            )
            continue  # Skip this problematic argument

        print(f"\n============================================================================")
        print(
            f"Processing Optimizer Argument: '{optimizer_file_argument}' (using base name: '{optimizer_config_filename_base}', resolved to load: '{actual_optimizer_config_filepath}')"
        )
        print(f"============================================================================\n")
        optimizer_config = load_config(actual_optimizer_config_filepath)
        # Process expressions in optimizer config using problem context
        optimizer_config = process_config_values(optimizer_config, context)

        # --- Per-Optimizer Config Setup ---
        if "optimizer_params" not in optimizer_config:
            raise ValueError(f"optimizer_params not found in optimizer config: {actual_optimizer_config_filepath}")
        optimizer_params = optimizer_config["optimizer_params"]

        if "device" not in optimizer_params:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            optimizer_params["device"] = default_device
            print(f"Device not specified in {actual_optimizer_config_filepath}, using default: {default_device}")

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

            # Name for WandB UI
            wandb_run_name = f"{optimizer_config_filename_base}"

            # Unique identifier for the completion log (includes problem, optimizer, params, and seed)
            completion_id_stable_string = config_to_stable_string(current_run_config)
            completion_id_hash = hashlib.md5(completion_id_stable_string.encode()).hexdigest()[
                :8
            ]  # Consistent hash length
            completion_run_id = (
                f"{optimizer_config_filename_base}_{completion_id_hash}_{seed}"  # Make seed explicit for clarity in log
            )

            print(
                f"\n--- Seed {seed}/{N_runs-1}: Checking run with completion ID {completion_run_id} for optimizer {optimizer_config_filename_base} ---"
            )

            if completion_manager.check_if_run_completed(completion_run_id):
                print(f"--- Skipping already completed run (ID: {completion_run_id}) ---")
                skipped_runs_count += 1
                continue

            wandb_run = None
            success = False
            try:
                print(
                    f"--- Starting run: {wandb_run_name} (Project: {project_name}, Group: {group_name}, Completion ID: {completion_run_id}) ---"
                )
                wandb_run = wandb.init(
                    entity="USNA",
                    project=project_name,
                    config=current_run_config,
                    group=group_name,
                    name=wandb_run_name,
                    mode="online",
                )

                run_experiment(problem_config, optimizer_config, seed, project_name)

                completion_manager.log_run_completion(completion_run_id, project_name)
                completed_runs_count += 1
                success = True
                print(f"--- Finished and logged run (ID: {completion_run_id}, WandB Name: {wandb_run_name}) ---")

            except Exception as e:
                print(
                    f"!!! ERROR during execution for {optimizer_config_filename_base}, seed {seed} (Run: {wandb_run_name}): {e} !!!"
                )
                traceback.print_exc()
                raise e

            finally:
                if wandb_run is not None:
                    exit_code = 0 if success else 1
                    wandb.finish(exit_code=exit_code)
                    print(f"--- WandB run finished for {wandb_run_name} (Exit code: {exit_code}) ---")

        print(f"\n--- Summary for Optimizer {optimizer_config_filename_base} ---")
        print(f"Total seeds requested: {N_runs}")
        print(f"Runs skipped (already complete): {skipped_runs_count}")
        print(f"Runs completed successfully: {completed_runs_count}")
        print(f"----------------------------------------------------------------------------")

    # --- Overall Summary  ---
    print(f"\n--- All optimizer configurations processed. ---")
