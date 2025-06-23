import time
import wandb
import math
from tqdm import tqdm
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader, IterableDataset

from objective_functions import LinearRegression, LogisticRegression, BaseObjectiveFunction
from optimizers import SGD, mSNA, SNA, BaseOptimizer
from datasets import generate_regression, load_dataset_from_source


def get_optimizer_class(optimizer_name: str) -> type[BaseOptimizer]:
    if optimizer_name == "SGD":
        return SGD
    elif optimizer_name == "mSNA":
        return mSNA
    elif optimizer_name == "SNA":
        return SNA
    else:
        raise ValueError(f"Unknown optimizer specified in config: {optimizer_name}")


def get_obj_function_class(model_type: str) -> Any:
    model_type_lower = model_type.lower()
    if model_type_lower == "linear_regression":
        return LinearRegression
    elif model_type_lower == "logistic_regression":
        return LogisticRegression
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# --- Helper Function for Evaluation ---
def evaluate_on_set(
    eval_set: torch.utils.data.Dataset,
    model: BaseObjectiveFunction,
    param: torch.Tensor,
    device: str,
    model_name: str,
    set_name: str,
    eval_batch_size: int = 512,
    subset_size: int | None = None,
) -> dict:
    """Helper to evaluate loss and accuracy on a given dataset or a subset of it."""
    if eval_set is None or len(eval_set) == 0:
        return {}

    sampler = None
    if subset_size and subset_size < len(eval_set):
        # Use a random subset of the data for evaluation
        indices = torch.randperm(len(eval_set))[:subset_size]
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        total_samples = subset_size
    else:
        total_samples = len(eval_set)

    eval_loader = DataLoader(eval_set, batch_size=eval_batch_size, pin_memory=(device == "cuda"), sampler=sampler)
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch_cpu, y_batch_cpu in eval_loader:
            X_batch, y_batch = X_batch_cpu.to(device), y_batch_cpu.to(device)
            loss = model((X_batch, y_batch), param)
            total_loss += loss.item() * X_batch.size(0)

            if model_name == "LogisticRegression":
                phi_batch = model._add_bias(X_batch)
                logits = torch.matmul(phi_batch, param)
                predictions = (torch.sigmoid(logits) > 0.5).float()
                all_predictions.append(predictions.cpu())
                all_targets.append(y_batch_cpu.squeeze())

    metrics = {}
    if total_samples > 0:
        metrics[f"{set_name}_loss"] = total_loss / total_samples
        if all_predictions:
            predictions_tensor = torch.cat(all_predictions)
            targets_tensor = torch.cat(all_targets)
            correct_predictions = (predictions_tensor == targets_tensor).sum().item()
            metrics[f"{set_name}_accuracy"] = correct_predictions / total_samples
    return metrics


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
    log_test_every_n_batches = problem_config.get("log_test_every_n_batches", 0)
    log_train_every_n_batches = problem_config.get("log_train_every_n_batches", 0)

    # --- Setup: Model ---
    # Instantiate the model early, as it's needed for Hessian estimation.
    model = get_obj_function_class(model_name)(**model_params)

    # --- Setup: Data, Model, Initial Params ---
    # Ensure global seed is set for other one-off random operations (e.g., theta_init)
    torch.manual_seed(seed)
    batch_size = optimizer_params["batch_size"]  # This is the data_batch_size for the iterable dataset

    param_dim: int
    train_set: torch.utils.data.Dataset
    val_set: torch.utils.data.Dataset | None = None  # Not used for now
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
            print(f"   Estimating true_hessian for {dataset_name} at true_theta as no close formula was provided...")

            # --- Hessian Estimation ---
            n_samples_for_hessian_estimation = int(1e6)
            # Use a large batch size for efficiency
            hessian_estimation_batch_size = 2048

            print(
                f"   Using a temporary dataset of {n_samples_for_hessian_estimation:.0e} samples (batch size: {hessian_estimation_batch_size}) for this estimation."
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

            number_of_batches = 0
            for X_batch_hess_cpu, y_batch_hess_cpu in hessian_estimation_loader:
                if X_batch_hess_cpu.shape[0] != hessian_estimation_batch_size:
                    break
                # Move data to the model device for the model's Hessian calculation
                X_batch_hess, y_batch_hess = X_batch_hess_cpu.to(device), y_batch_hess_cpu.to(device)

                batch_hess = model.hessian((X_batch_hess, y_batch_hess), true_theta)
                accumulated_hessian += batch_hess
                number_of_batches += 1

            if number_of_batches == 0:
                raise ValueError(
                    "Hessian estimation did not process any batches. "
                    "This is likely because n_samples_for_hessian_estimation is smaller than hessian_estimation_batch_size."
                )
            true_hessian = accumulated_hessian / number_of_batches
            print(f"   Finished estimating true_hessian. Averaged over {number_of_batches} batches.")

    else:
        print(f"   Loading real dataset: {dataset_name}...")
        loaded_data = load_dataset_from_source(dataset_name=dataset_name, random_state=seed, **dataset_params)
        train_set = loaded_data["train_dataset"]
        val_set = loaded_data["val_dataset"]
        test_set = loaded_data["test_dataset"]
        n_train_set = loaded_data["n_train"]
        n_test_set = loaded_data["n_test"]
        number_features = loaded_data["number_features"]

        true_theta, true_hessian = None, None
        bias = problem_config["model_params"].get("bias")
        param_dim = number_features + 1 if bias else number_features

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
    optimizer: BaseOptimizer
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

        inv_hess_error_fro = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord="fro").item() ** 2
        inv_hess_error_operator = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord=2).item()
        log_data_initial["inv_hess_error_fro"] = inv_hess_error_fro
        log_data_initial["inv_hess_error_operator"] = inv_hess_error_operator
        if compute_inv_hess_error_avg:
            inv_hess_error_fro_avg = torch.linalg.norm(true_inv_hessian - optimizer.matrix_avg, ord="fro").item() ** 2
            inv_hess_error_operator_avg = torch.linalg.norm(true_inv_hessian - optimizer.matrix_avg, ord=2).item()
            log_data_initial["inv_hess_error_fro_avg"] = inv_hess_error_fro_avg
            log_data_initial["inv_hess_error_operator_avg"] = inv_hess_error_operator_avg

    # Initial test set evaluation
    if test_set is not None:
        initial_test_metrics = evaluate_on_set(test_set, model, theta_init, device, model_name, set_name="test")
        log_data_initial.update(initial_test_metrics)

    wandb.log(log_data_initial)

    # --- Training Loop (No separate warm-up phase) ---
    # The dataloader will use the `train_set` which is already configured
    # with its own independent, seeded generator (`rng_train`). No further seeding is needed here.
    is_iterable_dataset = isinstance(train_set, torch.utils.data.IterableDataset)
    dataloader = DataLoader(
        train_set,
        batch_size=None if is_iterable_dataset else batch_size,
        shuffle=False,  # Keep to False for reproducibility across dataset types
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

    for batch_idx, data_batch_cpu in enumerate(progress_bar_iterator):
        # data_batch_cpu is a tensor or tuple of tensors on the CPU
        current_batch_size = (
            data_batch_cpu[0].size(0) if isinstance(data_batch_cpu, (list, tuple)) else data_batch_cpu.size(0)
        )

        current_step_duration: float
        if device == "cuda":
            # Time the data transfer and the optimizer step together for GPU
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            if isinstance(data_batch_cpu, (list, tuple)):
                data_on_device = tuple(item.to(device) for item in data_batch_cpu)
            else:
                data_on_device = data_batch_cpu.to(device)

            optimizer.step(data_on_device)
            end_event.record()

            torch.cuda.synchronize()
            current_step_duration = start_event.elapsed_time(end_event) / 1000.0
        else:  # CPU
            # For CPU, the .to(device) is a no-op but we include it for structural consistency.
            # The timing correctly captures just the optimizer step.
            step_start_cpu_time = time.perf_counter()

            if isinstance(data_batch_cpu, (list, tuple)):
                data_on_device = tuple(item.to(device) for item in data_batch_cpu)
            else:
                data_on_device = data_batch_cpu.to(device)

            optimizer.step(data_on_device)
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
            inv_hess_error_fro = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord="fro").item() ** 2
            inv_hess_error_operator = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord=2).item()
            log_data_step["inv_hess_error_fro"] = inv_hess_error_fro
            log_data_step["inv_hess_error_operator"] = inv_hess_error_operator

            if compute_inv_hess_error_avg:
                inv_hess_error_fro_avg = (
                    torch.linalg.norm(true_inv_hessian - optimizer.matrix_avg, ord="fro").item() ** 2
                )
                inv_hess_error_operator_avg = torch.linalg.norm(true_inv_hessian - optimizer.matrix_avg, ord=2).item()
                log_data_step["inv_hess_error_fro_avg"] = inv_hess_error_fro_avg
                log_data_step["inv_hess_error_operator_avg"] = inv_hess_error_operator_avg

        if hasattr(optimizer, "log_metrics"):
            for key, value in optimizer.log_metrics.items():
                log_data_step[f"opt_{key}"] = value

        # Periodic test set evaluation
        if log_test_every_n_batches > 0 and (batch_idx + 1) % log_test_every_n_batches == 0:
            if test_set is not None:
                test_metrics = evaluate_on_set(test_set, model, optimizer.param, device, model_name, set_name="test")
                log_data_step.update(test_metrics)

        # Periodic train set evaluation
        if log_train_every_n_batches > 0 and (batch_idx + 1) % log_train_every_n_batches == 0:
            if train_set is not None:
                # Use a subset of the train set for faster evaluation
                train_eval_metrics = evaluate_on_set(
                    train_set,
                    model,
                    optimizer.param,
                    device,
                    model_name,
                    set_name="train",
                    subset_size=int(1e5),
                )
                log_data_step.update(train_eval_metrics)

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
            final_test_metrics = evaluate_on_set(test_set, model, optimizer.param, device, model_name, set_name="test")
            for key, value in final_test_metrics.items():
                # Strip prefix for cleaner final print summary
                metric_name = key.replace("test_", "")
                print(f"   Test Set {metric_name.capitalize()}: {value:.4f}")
                log_data_final[key] = value
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
