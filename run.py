import time
import wandb
import math
from tqdm import tqdm
from typing import Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset

from datasets import generate_regression, load_dataset_from_source
from utils import evaluate_on_set, get_obj_function_class, get_optimizer_class
from objective_functions import BaseObjectiveFunction
from optimizers import BaseOptimizer, mSNA


def initialize_theta(
    theta_init: torch.Tensor,
    initialization_batch: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    obj_function: BaseObjectiveFunction,
    gd_lr: float,
    device: str,
) -> None:
    """
    Performs deterministic Gradient Descent on a large initial dataset
    to find a better starting point (theta_init) for the main optimization.
    """
    NB_GD_STEPS = 100  # Fixed number of steps for now

    # The batch size for the print statement can be inferred from the data
    batch_size = (
        initialization_batch[0].shape[0]
        if isinstance(initialization_batch, (list, tuple))
        else initialization_batch.shape[0]
    )

    print(
        f"   [Initializer] Performing {NB_GD_STEPS} steps of deterministic GD with lr={gd_lr} on a large batch of size {batch_size}"
    )

    # Move data to the model device for the model's Hessian calculation
    if isinstance(initialization_batch, (list, tuple)):
        initialization_batch_on_device = tuple(item.to(device, non_blocking=True) for item in initialization_batch)
    else:
        initialization_batch_on_device = initialization_batch.to(device, non_blocking=True)

    for step in range(NB_GD_STEPS):
        grad = obj_function.grad(initialization_batch_on_device, theta_init)
        theta_init.add_(grad, alpha=-gd_lr)
        # Optional: Log the loss every 10 steps
        if (step + 1) % 10 == 0:
            with torch.no_grad():
                loss = obj_function(initialization_batch_on_device, theta_init)
                wandb.log({"initializer_loss": loss.item()})

    print("   [Initializer] GD initialization complete.")


def run_experiment(problem_config: dict, optimizer_config: dict, seed: int) -> None:
    """
    Runs the core optimization loop for a single experiment seed.
    Assumes wandb is already initialized.

    Args:
        problem_config (dict): The configuration dictionary for the problem.
        optimizer_config (dict): The configuration dictionary for the optimizer.
        seed (int): The seed for the experiment.
    """

    # Set the global seed for reproducibility
    torch.manual_seed(seed)

    # --- Extract Parameters ---
    optimizer_params = optimizer_config.get("optimizer_params", {})
    init_params = optimizer_config.get("init_params", {})
    device = optimizer_params.get("device")  # Defined in main
    radius = problem_config.get("radius")
    optimizer_name = optimizer_config.get("optimizer")
    dataset_name = problem_config.get("dataset")
    dataset_params = problem_config.get("dataset_params")
    model_name = problem_config.get("model")
    model_params = problem_config.get("model_params")
    log_test_every_n_batches = problem_config.get("log_test_every_n_batches", 0)
    log_train_every_n_batches = problem_config.get("log_train_every_n_batches", 0)
    optimal_lr = problem_config.get("optimal_lr")

    # --- Setup: Objective Function ---
    obj_function = get_obj_function_class(model_name)(**model_params)

    # --- Setup: Data and param_dim---
    train_set: torch.utils.data.Dataset
    test_set: torch.utils.data.Dataset | None  # No test set for synthetic datasets
    param_dim: int

    batch_size = optimizer_params["batch_size"]

    if "synthetic" in dataset_name:  # Synthetic datasets are generated on the fly
        # Define a device for the data generation process itself.
        data_gen_device = device if "cuda" in str(device) else "cpu"

        train_set, true_theta, true_hessian = generate_regression(
            dataset_name=dataset_name,
            dataset_params=dataset_params,
            device=device,
            data_batch_size=batch_size,
            seed=seed,
            data_gen_device=data_gen_device,
        )
        param_dim = true_theta.shape[0]
        n_train_set = dataset_params.get("n_dataset")

        if true_hessian is None and true_theta is not None:
            print(f"   Estimating true_hessian for {dataset_name} at true_theta as no close formula was provided...")

            n_samples_for_hessian_estimation = int(1e6)
            hessian_estimation_batch_size = 2048  # Large batch size for efficiency

            print(
                f"   Using a temporary dataset of {n_samples_for_hessian_estimation:.0e} samples (batch size: {hessian_estimation_batch_size}) for this estimation."
            )

            # Create a temporary config for the Hessian dataset with the capped sample size
            dataset_params_hessian = dataset_params.copy()
            dataset_params_hessian["n_dataset"] = n_samples_for_hessian_estimation

            # Create and seed a separate, identically-seeded dataset for Hessian estimation
            hessian_dataset, _, _ = generate_regression(
                dataset_name=dataset_name,
                dataset_params=dataset_params_hessian,
                device=device,
                data_batch_size=hessian_estimation_batch_size,
                seed=seed,
                data_gen_device=data_gen_device,
            )

            accumulated_hessian = torch.zeros((param_dim, param_dim), device=device, dtype=true_theta.dtype)

            hessian_estimation_loader = DataLoader(
                hessian_dataset,
                batch_size=None,  # Pass batches from dataset as-is
                shuffle=False,
                pin_memory=(device == "cuda"),
            )

            number_of_batches = 0.0
            for data_batch_cpu in hessian_estimation_loader:
                current_batch_size = (
                    data_batch_cpu[0].size(0) if isinstance(data_batch_cpu, (list, tuple)) else data_batch_cpu.size(0)
                )
                if current_batch_size == hessian_estimation_batch_size:
                    weight = 1
                else:
                    weight = float(current_batch_size) / hessian_estimation_batch_size

                # Move data to the model device for the model's Hessian calculation
                if isinstance(data_batch_cpu, (list, tuple)):
                    data_on_device = tuple(item.to(device, non_blocking=True) for item in data_batch_cpu)
                else:
                    data_on_device = data_batch_cpu.to(device, non_blocking=True)

                batch_hess = obj_function.hessian(data_on_device, true_theta)
                accumulated_hessian += batch_hess * weight
                number_of_batches += weight

            true_hessian = accumulated_hessian / number_of_batches
            print(f"   Finished estimating true_hessian. Averaged over {number_of_batches} batches.")
            del hessian_dataset, hessian_estimation_loader

        if init_params.get("init_hess_inv", False):
            initialization_batch_size = int(init_params.get("init_hess_inv_samples"))
            print(f"   Initializing the optimizer matrix on a first batch of size {initialization_batch_size}")

            # Create a temporary config for the initialization dataset
            dataset_params_init = dataset_params.copy()
            dataset_params_init["n_dataset"] = initialization_batch_size

            # Generate a dedicated dataset for Hessian initialization
            init_dataset, _, _ = generate_regression(
                dataset_name=dataset_name,
                dataset_params=dataset_params_init,
                device=device,
                data_batch_size=initialization_batch_size,  # Generate as a single batch
                seed=seed,  # Use the same seed for the initialization batch
                data_gen_device=data_gen_device,
            )

            # Get the single batch of data from the dataset
            initialization_batch = next(iter(init_dataset))
            del init_dataset
            print(f"   Finished generating initialization batch of size {initialization_batch[0].shape[0]}")

    else:
        true_theta, true_hessian = None, None

        print(f"   Loading real dataset: {dataset_name}...")
        loaded_data = load_dataset_from_source(dataset_name=dataset_name, random_state=seed, **dataset_params)
        initialization_batch = loaded_data["initialization_batch"]
        train_set = loaded_data["train_dataset"]
        test_set = loaded_data["test_dataset"]
        n_train_set = loaded_data["n_train"]

        param_dim = loaded_data["number_features"] + 1 if model_params.get("bias") else loaded_data["number_features"]
    # --- End Setup: Data and param_dim ---

    # --- Define theta_init ---
    if "synthetic" in dataset_name:  # no theta initialization for synthetic datasets
        print(f"Initializing theta_init on a sphere of radius {radius} around true_theta")
        random_direction = torch.randn_like(true_theta)
        random_direction /= torch.linalg.vector_norm(random_direction)
        theta_init = true_theta + radius * random_direction
    else:  # real dataset, initialize to zeros or with GD
        theta_init = torch.zeros(param_dim, device=device, dtype=torch.float32)
        if problem_config.get("init_theta", False):
            # Also check if there is data to initialize on
            if initialization_batch and initialization_batch[0] is not None:
                initialize_theta(theta_init, initialization_batch, obj_function, optimal_lr, device=device)
            else:
                print("   [Initializer] Skipping GD initialization because the initialization dataset is empty.")
        else:
            print(f"Initializing theta_init to zeros for real dataset (param_dim: {param_dim})")

    # --- Start timers ---
    start_time = time.time()
    optimizer_time_cumulative = 0.0

    # --- Initialize Optimizer ---
    optimizer_setup_duration: float
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        optimizer_class = get_optimizer_class(optimizer_name)
        optimizer = optimizer_class(param=theta_init, obj_function=obj_function, **optimizer_params)

        if init_params.get("init_hess_inv", False):
            if initialization_batch and initialization_batch[0] is not None:
                # Move the initialization batch to the correct device before using it.
                init_data_on_device = tuple(item.to(device, non_blocking=True) for item in initialization_batch)
                regularization = init_params.get("init_hess_inv_reg", BaseOptimizer.DEFAULT_REGULARIZATION)
                optimizer.initialize_hessian(init_data_on_device, regularization=regularization)
                del init_data_on_device
            else:
                print("   [Optimizer] Skipping Hessian initialization because the initialization dataset is empty.")

        end_event.record()
        torch.cuda.synchronize()
        optimizer_setup_duration = start_event.elapsed_time(end_event) / 1000.0
    else:  # CPU
        optimizer_setup_start_time = time.perf_counter()

        optimizer_class = get_optimizer_class(optimizer_name)
        optimizer = optimizer_class(param=theta_init, obj_function=obj_function, **optimizer_params)

        if init_params.get("init_hess_inv", False):
            if initialization_batch and initialization_batch[0] is not None:
                # For CPU, .to(device) is a no-op, but we handle it as a tuple for consistency.
                init_data_on_device = tuple(item.to(device) for item in initialization_batch)
                regularization = init_params.get("init_hess_inv_reg", BaseOptimizer.DEFAULT_REGULARIZATION)
                optimizer.initialize_hessian(init_data_on_device, regularization=regularization)
                del init_data_on_device
            else:
                print("   [Optimizer] Skipping Hessian initialization because the initialization dataset is empty.")

        optimizer_setup_end_time = time.perf_counter()
        optimizer_setup_duration = optimizer_setup_end_time - optimizer_setup_start_time

    # The initialization batch is no longer needed after this point.
    if "initialization_batch" in locals():
        del initialization_batch

    optimizer_time_cumulative += optimizer_setup_duration
    print(f"   Optimizer setup and Hessian initialization took: {optimizer_setup_duration:.4f}s")

    # Adjust lr_add based on optimal_lr setting in problem config, so that step sizes are not too large
    if optimal_lr is not None and optimal_lr > 0:
        original_lr_add = optimizer.lr_add
        optimizer.lr_add = max((1 / optimal_lr) - 1, original_lr_add)
        print(f"   [LR Adjust] Overriding optimizer's lr_add. Old: {original_lr_add:.4f}, New: {optimizer.lr_add:.4f}")

        # For mSNA, synchronize lr_hess_add with the new lr_add
        if isinstance(optimizer, mSNA):
            original_lr_hess_add = optimizer.lr_hess_add
            optimizer.lr_hess_add = optimizer.lr_add
            print(
                f"   [LR Adjust] Syncing mSNA's lr_hess_add. Old: {original_lr_hess_add:.4f}, New: {optimizer.lr_hess_add:.4f}"
            )

    # --- End Initialize Optimizer ---

    # --- Setup: Error Computation ---
    compute_theta_error = true_theta is not None
    compute_inv_hess_error = true_hessian is not None and hasattr(optimizer, "matrix")
    compute_inv_hess_error_avg = true_hessian is not None and getattr(optimizer, "averaged_matrix", False)

    if compute_inv_hess_error:
        true_inv_hessian = torch.linalg.inv(true_hessian)
        del true_hessian  # free memory

        # Optional: Calculate and print eigenvalues of the true inverse Hessian
        inv_hess_eigenvalues = torch.linalg.eigvalsh(true_inv_hessian)
        lambda_min_inv_hess = inv_hess_eigenvalues.min().item()
        lambda_max_inv_hess = inv_hess_eigenvalues.max().item()
        print(f"   True Inv Hessian: lambda_min={lambda_min_inv_hess:.4f}, lambda_max={lambda_max_inv_hess:.4f}")

    def compute_error(optimizer, log_dict):
        log_dict["estimation_error"] = torch.linalg.vector_norm(theta_init - true_theta).item() ** 2
        if compute_inv_hess_error:
            log_dict["inv_hess_error_fro"] = (
                torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord="fro").item() ** 2
            )
            log_dict["inv_hess_error_operator"] = torch.linalg.norm(true_inv_hessian - optimizer.matrix, ord=2).item()
        if compute_inv_hess_error_avg:
            log_dict["inv_hess_error_fro_avg"] = (
                torch.linalg.norm(true_inv_hessian - optimizer.matrix_avg, ord="fro").item() ** 2
            )
            log_dict["inv_hess_error_operator_avg"] = torch.linalg.norm(
                true_inv_hessian - optimizer.matrix_avg, ord=2
            ).item()

    # --- End Setup: Error Computation ---

    # --- Initial State Logging ---
    log_data_initial = {
        "samples": 1,  # Start samples at 1 for log scale compatibility
        "time": 0.0,
        "optimizer_time_cumulative": optimizer_time_cumulative,
    }
    if compute_theta_error:
        compute_error(optimizer, log_data_initial)
    else:
        log_data_initial.update(
            evaluate_on_set(train_set, obj_function, optimizer.param, device, model_name, set_name="train")
        )
        log_data_initial.update(
            evaluate_on_set(test_set, obj_function, optimizer.param, device, model_name, set_name="test")
        )
    wandb.log(log_data_initial)

    # --- Training Loop ---
    is_iterable_dataset = isinstance(train_set, IterableDataset)
    dataloader = DataLoader(
        train_set,
        batch_size=None if is_iterable_dataset else batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    data_iterator = iter(dataloader)
    cumulative_optimizer_samples = 0

    total_batches_in_dataset = math.ceil(n_train_set / batch_size)

    print(f"   Starting optimization loop for {total_batches_in_dataset} steps...")

    progress_bar_iterator = tqdm(
        data_iterator, total=total_batches_in_dataset, desc="   Optimization", unit="batch", leave=True
    )

    for batch_idx, data_batch_cpu in enumerate(progress_bar_iterator):
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
                data_on_device = tuple(item.to(device, non_blocking=True) for item in data_batch_cpu)
            else:
                data_on_device = data_batch_cpu.to(device, non_blocking=True)

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
        cumulative_optimizer_samples += current_batch_size

        loop_iteration_wall_time = time.time() - start_time

        # --- Log data for this step ---
        log_data_step = {
            "samples": 1 + cumulative_optimizer_samples,  # start at 1 for log scale compatibility
            "time": loop_iteration_wall_time,
            "optimizer_time_cumulative": optimizer_time_cumulative,
        }
        if compute_theta_error:
            compute_error(optimizer, log_data_step)
        else:
            # Periodic train set evaluation, on a subset of the train set for faster evaluation
            if log_train_every_n_batches > 0 and (batch_idx + 1) % log_train_every_n_batches == 0:
                train_eval_metrics = evaluate_on_set(
                    train_set,
                    obj_function,
                    optimizer.param,
                    device,
                    model_name,
                    set_name="train",
                    subset_size=int(5e5),
                )
                log_data_step.update(train_eval_metrics)

            # Periodic test set evaluation
            if log_test_every_n_batches > 0 and (batch_idx + 1) % log_test_every_n_batches == 0:
                test_eval_metrics = evaluate_on_set(
                    test_set, obj_function, optimizer.param, device, model_name, set_name="test"
                )
                log_data_step.update(test_eval_metrics)

        if hasattr(optimizer, "log_metrics"):  # If the optimizer has metrics to log
            for key, value in optimizer.log_metrics.items():
                log_data_step[f"opt_{key}"] = value

        wandb.log(log_data_step)
        # --- End Log data for this step ---

    final_wall_time = time.time() - start_time
    print(f"\n   Finished optimization loop. Total wall time: {final_wall_time:.2f}s")
    print(f"   Total optimizer step time: {optimizer_time_cumulative:.4f}s")

    # --- Final metrics ---
    log_data_final = {"optimizer_time": optimizer_time_cumulative}

    if compute_theta_error:  # Just print the final error for synthetic datasets
        final_error = torch.linalg.vector_norm(optimizer.param - true_theta).item() ** 2
        print(f"   Final estimation error: {final_error:.4f}")
    else:
        # For real datasets, evaluate on train and test sets
        print("   Calculating accuracy and loss on the train set...")
        final_train_metrics = evaluate_on_set(
            train_set, obj_function, optimizer.param, device, model_name, set_name="final_train"
        )
        log_data_final.update(final_train_metrics)
        for key, value in final_train_metrics.items():
            print(f"   {key.capitalize()}: {value:.4f}")

        print("   Calculating accuracy and loss on the test set...")
        final_test_metrics = evaluate_on_set(
            test_set, obj_function, optimizer.param, device, model_name, set_name="final_test"
        )
        log_data_final.update(final_test_metrics)
        for key, value in final_test_metrics.items():
            print(f"   {key.capitalize()}: {value:.4f}")

    # Log final optimizer metrics, if they exist
    if hasattr(optimizer, "log_metrics_end") and isinstance(optimizer.log_metrics_end, dict):
        log_data_final.update(optimizer.log_metrics_end)
    else:
        print("   No final optimizer metrics to log")

    wandb.log(log_data_final)
    # --- End Final metrics ---
