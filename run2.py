import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import wandb
import copy
from abc import ABC, abstractmethod
from torch.distributions import MultivariateNormal

# ============================================================================ #
# >>> Data Generator Classes <<<                                               #
# ============================================================================ #


class BaseDataGenerator(ABC):
    """Abstract base class for data generators.
    Handles seeding and defines the interface."""

    def __init__(self, seed, device):
        self.seed = seed
        self.device = device
        print(f"Generator initialized with seed {self.seed}")

    def _setup_seeds(self):
        """Set seeds for reproducibility right before generation if needed,
        or rely on the initial seed set externally."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        print(f"Generator seeds reset to {self.seed}")

    @abstractmethod
    def generate_chunk(self, chunk_size):
        """Generates a chunk of data."""
        pass

    @abstractmethod
    def get_metadata(self):
        """Returns a dictionary of metadata specific to this generator."""
        return {"seed": self.seed, "generator_class": self.__class__.__name__}

    def get_true_parameters(self):
        """Returns underlying true parameters, if any. Default is None."""
        return None


def _create_covariance_matrix(dim, cov_type, cov_param, device):
    """Helper function to create Sigma based on type and parameters."""
    print(f"Creating covariance matrix: type='{cov_type}', param='{cov_param}'")
    if cov_type == "identity":
        Sigma = torch.eye(dim, dtype=torch.float32, device=device)
    elif cov_type == "diagonal":
        if cov_param is None or len(cov_param) != dim:
            raise ValueError(
                f"Diagonal covariance requires 'cov_param' as a list/array of {dim} variances."
            )
        variances = torch.tensor(cov_param, dtype=torch.float32, device=device)
        if torch.any(variances <= 0):
            raise ValueError("Variances for diagonal covariance must be positive.")
        Sigma = torch.diag(variances)
    elif cov_type == "ar1":
        if not isinstance(cov_param, (float, int)) or abs(cov_param) >= 1:
            raise ValueError("AR1 covariance requires 'cov_param' (rho) between -1 and 1.")
        rho = cov_param
        i = torch.arange(dim, device=device)
        Sigma = rho ** torch.abs(i[:, None] - i)  # Calculate rho^|i-j|
    elif cov_type == "equicorrelation":
        if not isinstance(cov_param, (float, int)):
            raise ValueError("Equicorrelation requires 'cov_param' (rho).")
        rho = cov_param
        min_rho = -1.0 / (dim - 1) if dim > 1 else -1.0  # Handle dim=1 case
        if not (min_rho < rho < 1.0):
            raise ValueError(
                f"Equicorrelation parameter rho={rho} must be in the range"
                f" ({min_rho:.4f}, 1.0) for positive definiteness."
            )
        Sigma = torch.full((dim, dim), rho, dtype=torch.float32, device=device)
        Sigma.fill_diagonal_(1.0)
    else:
        raise ValueError(
            f"Unknown covariance_type: '{cov_type}'. Choose from 'identity', 'diagonal', 'ar1', 'equicorrelation'."
        )

    try:
        _ = torch.linalg.cholesky(Sigma)
        print(f"Successfully created positive definite Sigma matrix for type '{cov_type}'.")
    except torch.linalg.LinAlgError:
        print(
            f"Warning: Created Sigma matrix for type '{cov_type}' might not be positive definite."
        )

    return Sigma


class LinearModelGenerator(BaseDataGenerator):
    """Generates data for y = X_features*W (+ b) + noise.

    Args:
        dim (int): The number of features (dimension of X_features).
        bias (bool): Whether to include a bias term (b). If True, theta_true will have size dim+1.
                     If False, theta_true will have size dim.
        noise_std (float): Standard deviation of the Gaussian noise.
        covariance_type (str): Type of covariance matrix for X_features.
        covariance_param: Parameter for the covariance matrix (depends on type).
        theta_true_init: Optional initial true parameter for the model.
    """

    def __init__(
        self,
        seed,
        device,
        dim,
        bias: bool = True,
        noise_std=0.1,
        covariance_type="identity",
        covariance_param=None,
        theta_true_init: torch.Tensor | None = None,
    ):
        super().__init__(seed, device)
        if dim <= 0:
            raise ValueError("Dimension 'dim' (number of features) must be positive.")
        self.feature_dim = dim
        self.bias = bias
        self.total_dim = self.feature_dim + 1 if self.bias else self.feature_dim
        self.noise_std = noise_std
        self.covariance_type = covariance_type
        self.covariance_param = covariance_param

        self._setup_seeds()

        # Generate or use provided true parameter theta_true
        if theta_true_init is not None:
            expected_shape = (self.total_dim, 1)
            if theta_true_init.shape != expected_shape:
                raise ValueError(
                    f"Provided theta_true_init has shape {theta_true_init.shape}, "
                    f"but expected {expected_shape} based on feature_dim={self.feature_dim} and bias={self.bias}."
                )
            # Ensure tensor is on the correct device and dtype
            self.theta_true = theta_true_init.to(device=self.device, dtype=torch.float32)
            print("Using provided theta_true_init.")
        else:
            # If bias=True, theta_true = [W; b] where W is feature_dim x 1
            # If bias=False, theta_true = W where W is feature_dim x 1
            self.theta_true = torch.randn(
                self.total_dim, 1, dtype=torch.float32, device=self.device
            )
            print("Randomly generated theta_true.")

        self._setup_x_distribution()
        print(
            f"LinearModelGenerator initialized with feature_dim={self.feature_dim}, bias={self.bias}, "
            f"total_params={self.total_dim}, noise_std={noise_std}, cov_type='{covariance_type}'"
        )
        print(f"True parameter theta_true shape: {self.theta_true.shape}")

    def _setup_x_distribution(self):
        """Sets up the distribution specifically for generating X_features."""
        print(f"Setting up X feature distribution for {self.feature_dim} features...")
        mean_vector = torch.zeros(self.feature_dim, dtype=torch.float32, device=self.device)
        try:
            Sigma = _create_covariance_matrix(
                self.feature_dim, self.covariance_type, self.covariance_param, self.device
            )
            self.mvn_distribution_x_features = MultivariateNormal(
                loc=mean_vector, covariance_matrix=Sigma
            )
            print(f"Created MultivariateNormal for {self.feature_dim} features.")
        except (torch.linalg.LinAlgError, ValueError) as e:
            print(f"\n!!! Failed to create distribution for X features. Error: {e}\n")
            raise

    def generate_chunk(self, chunk_size):
        """Generates a chunk of features (X_features) and corresponding labels (y)."""
        X_features_chunk = self.mvn_distribution_x_features.sample((chunk_size,))
        noise = torch.randn(chunk_size, 1, dtype=torch.float32, device=self.device) * self.noise_std

        if self.bias:
            W_true = self.theta_true[:-1, :]  # Shape: (feature_dim, 1)
            b_true = self.theta_true[-1, :]  # Shape: (1,)
            y_chunk = torch.matmul(X_features_chunk, W_true) + b_true + noise  # b_true broadcasts
        else:
            W_true = self.theta_true  # Shape: (feature_dim, 1)
            y_chunk = torch.matmul(X_features_chunk, W_true) + noise

        return X_features_chunk, y_chunk

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.update(
            {
                "model_type": "linear_regression",
                "feature_dim": self.feature_dim,
                "bias": self.bias,
                "noise_std": self.noise_std,
                "covariance_type": self.covariance_type,
                "covariance_param": (
                    str(self.covariance_param) if self.covariance_param is not None else "None"
                ),
            }
        )
        return metadata

    def get_true_parameters(self):
        # Returns the true theta used in this generator
        return {"theta_true": self.theta_true}


# ============================================================================ #
# >>> Model Definition  <<<                                                    #
# ============================================================================ #

class BaseObjectiveFunction(ABC):
    """
    Abstract base class for different objective functions
    """

    

# ============================================================================ #
# >>> Configuration <<<                                                        #
# ============================================================================ #
config = {
    "project_name": "sgd_comparison_linear_regression",
    "entity": "gsalle",
    # --- Data Generator Configuration ---
    "data_generator": {
        "class": "LinearModelGenerator",
        "params": {
            "dim": 9,
            "bias": True,
            "noise_std": 0.5,
            "covariance_type": "identity",
            "covariance_param": None,
        },
    },
    # --- Model Configuration ---
    "model": {
        "bias": True,
        "input_dim": 9,
        "output_dim": 1,
    },
    # --- Training Configuration ---
    "training": {
        "total_samples": 50000,
        "chunk_size": 100,
        "loss_function": "L2_Error",
    },
    # --- Optimizer Configurations ---
    "optimizer_configs": [
        {
            "name": "SGD_lr_0.01_mom_0.0",
            "type": "SGD",
            "lr": 0.01,
            "momentum": 0.0,
        },
        {
            "name": "SGD_lr_0.001_mom_0.9",
            "type": "SGD",
            "lr": 0.001,
            "momentum": 0.9,
        },
    ],
    # --- Simulation Configuration ---
    "simulation": {
        "num_runs": 5,
        "base_seed": 123,
    },
}

# --- Configuration Validation ---
model_cfg = config["model"]
gen_cfg = config["data_generator"]["params"]

model_input_dim = model_cfg["input_dim"]
model_has_bias = model_cfg["bias"]
gen_feature_dim = gen_cfg["dim"]
gen_has_bias = gen_cfg["bias"]

if model_has_bias != gen_has_bias:
    raise ValueError(
        f"Model (bias={model_has_bias}) must match generator (bias={gen_has_bias}) configuration."
    )

if model_input_dim != gen_feature_dim:
    raise ValueError(
        f"Model (input_dim={model_input_dim}) must match generator (feature_dim={gen_feature_dim})."
    )

# ============================================================================ #
# >>> Logging Configuration <<<                                                #
# ============================================================================ #

def calculate_l2_parameter_error(theta_hat: torch.Tensor, theta_true: torch.Tensor | None) -> float | None:
    """
    Calculates the L2 norm (Euclidean distance) between estimated and true parameters.

    Args:
        theta_hat (torch.Tensor): The estimated parameter vector.
        theta_true (torch.Tensor | None): The true parameter vector. If None, error cannot be calculated.

    Returns:
        float | None: The L2 parameter error, or None if theta_true is None.
    """
    if theta_true is None:
        raise ValueError("theta_true is None, cannot calculate parameter error.")
    if not isinstance(theta_hat, torch.Tensor) or not isinstance(theta_true, torch.Tensor):
        raise TypeError("Both theta_hat and theta_true must be torch Tensors.")
    if theta_hat.shape != theta_true.shape:
        # Attempt to reshape theta_hat if it's 1D and theta_true is 2D column vector
        if theta_hat.ndim == 1 and theta_true.ndim == 2 and theta_true.shape[1] == 1:
             theta_hat = theta_hat.view(-1, 1)
        elif theta_true.ndim == 1 and theta_hat.ndim == 2 and theta_hat.shape[1] == 1:
             theta_true = theta_true.view(-1, 1)
        else:
            print(f"Warning: Shape mismatch between theta_hat ({theta_hat.shape}) and theta_true ({theta_true.shape}). Cannot calculate parameter error accurately.")
            return None # Or raise error depending on strictness needed

    # Ensure parameters are on the same device
    theta_true = theta_true.to(theta_hat.device)

    # Calculate L2 error
    error_vector = theta_hat.detach() - theta_true
    l2_error = torch.linalg.norm(error_vector).item()
    return l2_error

def log_parameter_error(step: int, theta_hat: torch.Tensor, theta_true: torch.Tensor | None, wandb_prefix: str = "") -> None:
    """
    Calculates and logs the L2 parameter error to WandB.

    Args:
        step (int): The current step (e.g., sample count) for logging.
        theta_hat (torch.Tensor): The estimated parameter vector.
        theta_true (torch.Tensor | None): The true parameter vector.
        wandb_prefix (str): Optional prefix for the WandB metric name (e.g., "Train/", "Eval/").
    """
    l2_error = calculate_l2_parameter_error(theta_hat, theta_true)

    if wandb.run is not None:
        metric_name = f"{wandb_prefix}L2_Parameter_Error"
        wandb.log({metric_name: l2_error}, step=step)
    else:
        raise ValueError("WandB run not active.")

def _get_theta_hat(model: nn.Module, model_has_bias: bool) -> torch.Tensor:
    """
    Extracts the parameter vector theta_hat from a LinearRegression model.

    Assumes the model has a 'linear' layer with 'weight' and potentially 'bias'.

    Args:
        model (nn.Module): The LinearRegression model instance.
        model_has_bias (bool): Whether the model includes a bias term according to configuration.

    Returns:
        torch.Tensor: The concatenated parameter vector theta_hat [W; b] or just [W].

    Raises:
        TypeError: If the provided model doesn't seem to be the expected LinearRegression.
        AttributeError: If the model lacks the 'linear' layer or 'weight'.
        ValueError: If the model configuration regarding bias mismatches the layer's state.
    """
    # Basic check for the expected structure
    if not hasattr(model, 'linear') or not hasattr(model.linear, 'weight'):
        raise TypeError("Parameter extraction helper expects a model with a 'linear' layer having 'weight'.")

    # Extract weight (and transpose)
    # .data is used to get the tensor without autograd history, similar to detach()
    # Transpose to match the typical column vector format [features, 1]
    weight_hat = model.linear.weight.data.T

    if model_has_bias:
        if model.linear.bias is None:
             raise ValueError("Model config indicates bias=True, but model.linear.bias is None.")
        # Reshape bias to be a column vector [1, 1]
        bias_hat = model.linear.bias.data.view(-1, 1)
        # Concatenate weight and bias
        theta_hat = torch.cat((weight_hat, bias_hat), dim=0)
    else:
        # If model is configured without bias, but the layer has one, maybe warn?
        if model.linear.bias is not None:
            print("Warning: Model config indicates bias=False, but model.linear.bias parameter exists.")
        # Use only the weights
        theta_hat = weight_hat

    return theta_hat


# >>> Training Loop Function <<<                                               #
# ============================================================================ #
def train_model(run_config, data_generator, seed):
    """
    Trains a model using streaming data, logging L2 parameter error.

    Args:
        run_config (dict): Dictionary containing parameters specific to this run.
        data_generator (BaseDataGenerator): Instantiated data generator object.
        seed (int): Random seed for this specific training run (model init, etc.).

    Returns:
        None: Metrics are logged to WandB.
    """
    # --- Reproducibility for this specific training run ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Initialize WandB Run ---
    run = wandb.init(
        project=config["project_name"],
        entity=config["entity"],
        config=run_config,
        group=run_config["optimizer"]["name"],
        job_type="train",
        name=f"run_{seed}",
    )
    # Log data generator metadata and true parameters (if available)
    generator_metadata = data_generator.get_metadata()
    true_params_dict = data_generator.get_true_parameters()
    wandb.config.update({"data_generator_metadata": generator_metadata})

    theta_true = None
    if true_params_dict and "theta_true" in true_params_dict:
        theta_true = true_params_dict["theta_true"]
        # wandb.config.update({"theta_true_vector": theta_true.cpu().numpy()}) # Optional: Log true vector
        print("Retrieved theta_true from generator.")
    else:
        print(
            "Warning: True parameters (theta_true) not found in generator. Cannot compute parameter error."
        )
        # Optionally raise an error if theta_true is essential
        # raise ValueError("theta_true not provided by the data generator.")

    # --- Setup Device, Model, Loss, Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run {seed} using device: {device}")

    # Move theta_true to the correct device if it exists
    if theta_true is not None:
        theta_true = theta_true.to(device)

    # Instantiate the model
    model = LinearRegression(config["model"]["input_dim"], config["model"]["output_dim"]).to(device)

    # Define loss function
    if config["training"]["loss_function"] == "MSELoss":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {config['training']['loss_function']}")

    # Select and configure optimizer
    opt_details = run_config["optimizer"]
    if opt_details["type"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=opt_details["lr"], momentum=opt_details.get("momentum", 0.0)
        )
    elif opt_details["type"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=opt_details["lr"], betas=opt_details.get("betas", (0.9, 0.999))
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_details['type']}")

    # WandB Watch (optional)
    wandb.watch(model, log="gradients", log_freq=run_config["log_freq"])

    # --- Streaming Training Loop ---
    print(f"Starting training run {seed} with optimizer: {run_config['optimizer']['name']}")
    total_samples = run_config["total_samples"]
    chunk_size = run_config["chunk_size"]
    num_chunks = (total_samples + chunk_size - 1) // chunk_size

    samples_processed = 0
    for chunk_idx in range(num_chunks):
        X_chunk, y_chunk = data_generator.generate_chunk(chunk_size)
        X_chunk, y_chunk = X_chunk.to(device), y_chunk.to(device)

        current_chunk_actual_size = X_chunk.shape[0]
        if samples_processed + current_chunk_actual_size > total_samples:
            limit = total_samples - samples_processed
            X_chunk = X_chunk[:limit]
            y_chunk = y_chunk[:limit]
            current_chunk_actual_size = limit

        if current_chunk_actual_size == 0:
            break

        # --- Training Step on Chunk ---
        model.train()
        optimizer.zero_grad()
        outputs = model(X_chunk)
        loss = loss_fn(outputs, y_chunk)
        loss.backward()
        optimizer.step()

        train_loss_chunk = loss.item()
        samples_processed += current_chunk_actual_size

        # --- Parameter Error Calculation & Logging (Periodically) ---
        if (chunk_idx + 1) % run_config["log_freq"] == 0 or samples_processed >= total_samples:
            param_error_l2 = float("nan")  # Default if theta_true is missing
            if theta_true is not None:
                model.eval()  # Ensure model params are used correctly (dropout/batchnorm off)
                with torch.no_grad():
                    # Get estimated parameters (weights W_hat and bias b_hat)
                    theta_hat_W = (
                        model.linear.weight.data
                    )  # Shape: (output_dim, input_dim) -> (1, feature_dim)
                    theta_hat_b = model.linear.bias.data  # Shape: (output_dim,) -> (1,)

                    # Combine estimated parameters into a single vector matching theta_true's structure [W; b]
                    # theta_hat_W is (1, feature_dim), need to transpose or squeeze
                    # theta_hat_b is (1,), need to reshape to (1, 1) for concat if W is (feature_dim, 1)
                    # Let's flatten both theta_hat and theta_true for easier comparison
                    theta_hat_W_flat = theta_hat_W.squeeze()  # Shape: (feature_dim,)
                    theta_hat_b_flat = theta_hat_b.squeeze().unsqueeze(0)  # Shape: (1,)
                    theta_hat_combined = torch.cat(
                        (theta_hat_W_flat, theta_hat_b_flat), dim=0
                    )  # Shape: (dim,)

                    # Ensure theta_true is also flat
                    theta_true_flat = theta_true.squeeze()  # Shape: (dim,)

                    # Calculate L2 norm of the difference vector
                    error_vec = theta_hat_combined - theta_true_flat
                    param_error_l2 = torch.linalg.norm(error_vec).item()

            # Log metrics against samples_processed
            wandb.log(
                {
                    "samples_processed": samples_processed,
                    "train_loss_chunk": train_loss_chunk,
                    "param_error_l2": param_error_l2,  # Log the L2 parameter error
                }
            )
            print(
                f"Samples: {samples_processed}/{total_samples} | Chunk {chunk_idx+1}/{num_chunks} | Train Loss (Chunk): {train_loss_chunk:.4f} | Param Error L2: {param_error_l2:.4f}"
            )

        if samples_processed >= total_samples:
            break

    # --- Finish WandB Run ---
    print(f"Finished training run {seed} after processing {samples_processed} samples.")
    run.finish()


# ============================================================================ #
# >>> Main Experiment Execution (No Validation Set) <<<                      #
# ============================================================================ #
if __name__ == "__main__":
    # --- Instantiate Data Generator ---
    gen_config = config["data_generator"]
    gen_params = copy.deepcopy(
        gen_config["params"]
    )  # Use deepcopy to avoid modifying original config
    device_for_data = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare theta_true_init if provided in config
    theta_true_init_value = gen_params.pop("theta_true_init", None)  # Remove from params dict
    theta_true_tensor = None
    if theta_true_init_value is not None:
        try:
            # Determine expected dimension
            feature_dim = gen_params["dim"]
            bias = gen_params.get("bias", True)  # Match default bias of generator
            total_dim = feature_dim + 1 if bias else feature_dim
            expected_len = total_dim

            if len(theta_true_init_value) != expected_len:
                raise ValueError(
                    f"Config 'theta_true_init' has length {len(theta_true_init_value)}, expected {expected_len}."
                )

            # Convert list from config to tensor [total_dim, 1]
            theta_true_tensor = torch.tensor(theta_true_init_value, dtype=torch.float32).unsqueeze(
                1
            )
            print(f"Prepared theta_true_init tensor with shape {theta_true_tensor.shape}")
        except Exception as e:
            print(f"Error processing 'theta_true_init' from config: {e}")
            print("Will proceed with random theta_true generation.")
            theta_true_tensor = None  # Ensure it's None if conversion fails

    # Instantiate the generator, passing the prepared tensor if available
    if gen_config["class"] == "LinearModelGenerator":
        data_generator = LinearModelGenerator(
            device=device_for_data,
            theta_true_init=theta_true_tensor,  # Pass the tensor here
            **gen_params,  # Pass remaining params like dim, bias, noise_std etc.
        )
    else:
        raise ValueError(f"Unknown data generator class: {gen_config['class']}")

    print(
        f"Instantiated data generator: {gen_config['class']}"
    )  # Seed info is printed inside generator now

    # --- Generate Fixed Validation Set --- >> REMOVED <<
    # No longer generating X_val, y_val

    # --- Iterate Through Optimizer Configurations and Training Seeds ---
    for opt_config in config["optimizer_configs"]:
        print(f"\n=== Running Experiments for Optimizer: {opt_config['name']} ===")

        for i in range(config["simulation"]["num_runs"]):
            run_seed = config["simulation"]["base_seed"] + i + 1
            print(
                f"--- Starting Run {i+1}/{config['simulation']['num_runs']} (Training Seed: {run_seed}) ---"
            )

            current_run_config = {
                "training_seed": run_seed,
                "optimizer": copy.deepcopy(opt_config),
                "total_samples": config["training"]["total_samples"],
                "chunk_size": config["training"]["chunk_size"],
                "log_freq": max(
                    1,
                    config["training"]["total_samples"] // (config["training"]["chunk_size"] * 20),
                ),
                "model_config": config["model"],
            }

            # Call the training function (without X_val, y_val)
            train_model(run_config=current_run_config, data_generator=data_generator, seed=run_seed)

    print("\nAll experiments finished. Check WandB project for results.")


# ============================================================================ #
# >>> Analysis and Plotting (Adapted for L2 Parameter Error) <<<             #
# ============================================================================ #
def analyze_results(project_name, entity):
    """
    Fetches run data from WandB, aggregates results, and plots L2 parameter error.
    """
    print("\n--- Analyzing Results from WandB ---")
    if entity == "your_wandb_username_or_team":
        print("Error: Still using placeholder 'your_wandb_username_or_team' in config.")
        print("Please replace it with your actual WandB username or team name and rerun.")
        return

    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project_name}")
    except Exception as e:
        print(f"Error fetching runs from WandB: {e}")
        return

    if not runs:
        print("No runs found.")
        return

    history_list = []
    for run in runs:
        # Fetch relevant keys including the new parameter error metric
        run_history = run.history(
            keys=["samples_processed", "param_error_l2", "train_loss_chunk", "_step"]
        )
        run_history["group"] = run.group
        run_history["training_seed"] = run.config.get("training_seed", "unknown")
        run_history["optimizer_type"] = run.config.get("optimizer", {}).get("type", "unknown")
        run_history["optimizer_lr"] = run.config.get("optimizer", {}).get("lr", "unknown")
        history_list.append(run_history)

    if not history_list:
        print("No history data found.")
        return

    history_df = pd.concat(history_list, ignore_index=True)

    # --- Data Cleaning ---
    history_df["samples_processed"] = pd.to_numeric(
        history_df["samples_processed"], errors="coerce"
    )
    history_df["param_error_l2"] = pd.to_numeric(history_df["param_error_l2"], errors="coerce")
    history_df["train_loss_chunk"] = pd.to_numeric(history_df["train_loss_chunk"], errors="coerce")
    history_df.dropna(
        subset=["samples_processed", "param_error_l2", "group"], inplace=True
    )  # Ensure key metric is present

    if history_df.empty:
        print("No valid data points found after cleaning.")
        return

    print(f"Analyzing {len(history_df)} data points from {len(runs)} runs.")

    # --- Plotting Average L2 Parameter Error ---
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=history_df,
        x="samples_processed",
        y="param_error_l2",  # Plot the parameter error
        hue="group",
        errorbar=("sd", 1),
        legend="full",
    )
    plt.title("Average L2 Parameter Error vs Samples Processed (Mean ± SD across runs)")
    plt.xlabel("Samples Processed")
    plt.ylabel("L2 Norm of Parameter Error (||theta_hat - theta_true||)")
    plt.yscale("log")  # Often useful for error plots
    plt.legend(title="Optimizer Config", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # --- Optional: Plot Training Loss (Chunk) ---
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=history_df,
        x="samples_processed",
        y="train_loss_chunk",
        hue="group",
        errorbar=("sd", 1),
        legend="full",
    )
    plt.title("Average Training Loss (Chunk) vs Samples Processed (Mean ± SD across runs)")
    plt.xlabel("Samples Processed")
    plt.ylabel("Training MSE Loss (Chunk)")
    plt.yscale("log")
    plt.legend(title="Optimizer Config", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # --- Optional: Print Summary Statistics of Final Error ---
    final_metrics_df = history_df.loc[
        history_df.groupby(["group", "training_seed"])["samples_processed"].idxmax()
    ]
    print("\n--- Summary of Final Metrics (Mean ± SD across runs) ---")
    # Include param_error_l2 in summary
    summary_stats = final_metrics_df.groupby("group")[["train_loss_chunk", "param_error_l2"]].agg(
        ["mean", "std"]
    )
    print(summary_stats)
