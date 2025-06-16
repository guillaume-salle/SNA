import torch
import torch.distributions as dist
from torch.utils.data import Dataset, IterableDataset
from typing import Generator, Tuple, Optional
import math
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import numpy as np


class MyDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        """
        Initialize the dataset.

        Parameters:
        X (torch.Tensor): Features of the dataset.
        Y (torch.Tensor, optional): Labels of the dataset. If None, the dataset consists only of features.
        """
        self.X = X
        self.Y = Y
        self.n_samples = len(X)

    def __iter__(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor, None, None]:
        """
        Make the dataset iterable, yielding either (x, y) tuples or just x depending on the presence of Y.
        """
        if self.Y is not None:
            for x, y in zip(self.X, self.Y):
                yield (x, y)
        else:
            for x in self.X:
                yield x

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Return the sample at the given index.

        Parameters:
        idx (int): Index of the sample to return.

        Returns:
        tuple or torch.Tensor: If Y is not None, return a tuple (x, y). Otherwise, return x.
        """
        if self.Y is not None:
            return (self.X[idx], self.Y[idx])
        else:
            return self.X[idx]


class RegressionIterableDataset(IterableDataset):
    def __init__(
        self,
        n_total_samples: int,
        true_theta: torch.Tensor,
        theta_dim: int,
        bias: bool,
        cov_matrix: torch.Tensor,
        data_batch_size: int,
        variance: float,
        dataset_name: str,
        rng_data: torch.Generator,
        data_gen_device: str,
    ):
        super().__init__()
        self.n_total_samples = n_total_samples
        self.bias = bias
        self.theta_dim = theta_dim
        self.cov_matrix = cov_matrix
        self.data_batch_size = data_batch_size
        self.variance = variance
        self.dataset_name = dataset_name
        self.rng_data = rng_data
        self.data_gen_device = data_gen_device

        self.feature_dim = self.theta_dim - 1 if self.bias else self.theta_dim

        # Move theta to the generation device once during initialization for efficiency
        self.true_theta_on_gen_device = true_theta.to(self.data_gen_device)

        # For manual sampling instead of using torch.distributions
        self.mean_features = None
        self.cholesky_L = None
        if self.feature_dim > 0:
            self.mean_features = torch.zeros(self.feature_dim, device=self.data_gen_device, dtype=torch.float32)
            # Pre-compute Cholesky decomposition for manual sampling
            self.cholesky_L = torch.linalg.cholesky(self.cov_matrix)

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        num_batches = math.ceil(self.n_total_samples / self.data_batch_size)
        samples_yielded_so_far = 0

        for _ in range(num_batches):
            current_actual_batch_size = min(self.data_batch_size, self.n_total_samples - samples_yielded_so_far)
            if current_actual_batch_size <= 0:
                break

            X_batch_gen: torch.Tensor
            if self.feature_dim > 0:
                if self.cholesky_L is not None:
                    # Manual sampling: z ~ N(0, I), then X = mean + z @ L.T
                    z = torch.randn(
                        current_actual_batch_size,
                        self.feature_dim,
                        device=self.data_gen_device,
                        generator=self.rng_data,
                    )
                    X_batch_gen = self.mean_features + torch.matmul(z, self.cholesky_L.T)
                else:
                    raise RuntimeError("Cholesky decomposition was not initialized though feature_dim > 0")
            else:
                X_batch_gen = torch.empty(
                    (current_actual_batch_size, 0), device=self.data_gen_device, dtype=torch.float32
                )

            phi_batch_gen: torch.Tensor
            if self.bias:
                ones_batch = torch.ones(
                    (current_actual_batch_size, 1), device=self.data_gen_device, dtype=X_batch_gen.dtype
                )
                phi_batch_gen = torch.cat([X_batch_gen, ones_batch], dim=1)
            else:
                phi_batch_gen = X_batch_gen

            Y_batch_gen: torch.Tensor
            if self.dataset_name == "synthetic_logistic_regression":
                logits = torch.matmul(phi_batch_gen, self.true_theta_on_gen_device)
                probabilities = torch.sigmoid(logits)
                Y_batch_gen = torch.bernoulli(probabilities, generator=self.rng_data).to(dtype=torch.float32)
            elif self.dataset_name == "synthetic_linear_regression":
                mean_for_y = torch.matmul(phi_batch_gen, self.true_theta_on_gen_device)
                Y_batch_gen = torch.normal(mean=mean_for_y, std=math.sqrt(self.variance), generator=self.rng_data)
            else:
                raise ValueError(f"Unknown problem_model_type: {self.dataset_name} in dataset generation")

            samples_yielded_so_far += current_actual_batch_size
            yield X_batch_gen.to("cpu"), Y_batch_gen.to("cpu")


def toeplitz_matrix(d: int, cov_const: float | None = None, diag: bool = False, device: str = "cpu") -> torch.Tensor:
    if cov_const is None:
        cov_const = 1.0 / d
    if d > 0 and cov_const <= 0:
        raise ValueError("cov_const must be positive for toeplitz_matrix when d > 0.")
    if d == 0:
        return torch.empty((0, 0), device=device, dtype=torch.float32)

    indices = torch.arange(d, device=device)
    abs_diff = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))
    log_cov_const = torch.log(torch.tensor(cov_const, device=device, dtype=torch.float32))
    log_matrix = abs_diff.float() * log_cov_const
    matrix = torch.exp(log_matrix)

    if diag:
        diag_values = (1.0 + indices).to(matrix.dtype)
        matrix.diagonal().copy_(diag_values)
    return matrix


def hard_matrix(d: int, cov_const: float | None = None, device: str = "cpu") -> torch.Tensor:
    if d % 2 != 0:
        raise ValueError("d must be even for hard matrix")
    if d == 0:
        return torch.empty((0, 0), device=device, dtype=torch.float32)
    if cov_const is None:
        cov_const = 100.0
    matrix = torch.zeros(d, d, device=device, dtype=torch.float32)
    block = torch.tensor([[1 / cov_const, 1], [1, cov_const + 1]], device=device, dtype=torch.float32)
    for i in range(d // 2):
        matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
    return matrix


def random_matrix(
    d: int, cov_const: float | None = None, device: str = "cpu", generator: torch.Generator | None = None
) -> torch.Tensor:
    if d == 0:
        return torch.empty((0, 0), device=device, dtype=torch.float32)
    if cov_const is None:
        cov_const = 1.0
    if cov_const <= 0:
        raise ValueError("cov_const (exponent) must be positive for this eigenvalue distribution.")

    A = torch.randn(d, d, device=device, dtype=torch.float32, generator=generator)
    Q, _ = torch.linalg.qr(A)
    Q = Q.to(dtype=torch.float32)

    j_values = torch.arange(1, d + 1, device=device, dtype=torch.float32)
    new_eigenvalues = 1.0 / (j_values**cov_const)
    new_eigenvalues, _ = torch.sort(new_eigenvalues)

    new_cov_matrix = Q @ torch.diag(new_eigenvalues) @ Q.T

    # print(f"Target eigenvalues for random_matrix: {new_eigenvalues}")  # Debug print
    eigenvalues = torch.linalg.eigvalsh(new_cov_matrix)
    min_eig = torch.min(eigenvalues)
    max_eig = torch.max(eigenvalues)
    print(f"    Min eigenvalue of the new matrix: {min_eig}, Max eigenvalue: {max_eig}")
    return new_cov_matrix


def generate_covariance_matrix(
    feature_dim: int,
    cov_type: str,
    cov_const: float | None = None,
    diag: bool = False,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if feature_dim < 0:
        raise ValueError(f"Feature dimension must be non-negative, got {feature_dim}")
    if cov_type not in ["identity", "toeplitz", "hard", "random"]:
        raise ValueError(f"Invalid covariance matrix type: {cov_type}")
    if cov_const is not None and cov_type != "identity" and cov_const <= 0:
        raise ValueError(f"Covariance constant must be positive for {cov_type}, got {cov_const}")

    if feature_dim == 0:
        return torch.empty((0, 0), device=device, dtype=torch.float32)

    try:
        if cov_type == "identity":
            return torch.eye(feature_dim, device=device, dtype=torch.float32)
        elif cov_type == "toeplitz":
            return toeplitz_matrix(feature_dim, cov_const=cov_const, diag=diag, device=device)
        elif cov_type == "hard":
            return hard_matrix(feature_dim, cov_const=cov_const, device=device)
        elif cov_type == "random":
            return random_matrix(feature_dim, cov_const=cov_const, device=device, generator=generator)
    except Exception as e:
        raise RuntimeError(f"Error generating covariance matrix of type '{cov_type}': {str(e)}")
    raise ValueError(f"Unhandled covariance matrix type: {cov_type}")


def generate_regression(
    dataset_name: str,
    dataset_params: dict,
    device: str,
    data_batch_size: int,
    rng_data: torch.Generator,
    data_gen_device: str,
) -> Tuple[RegressionIterableDataset, torch.Tensor, torch.Tensor | None]:
    """
    Generate data from a linear or logistic regression model using PyTorch, returning an iterable dataset.
    The dataset performs generation on `data_gen_device` but yields CPU tensors.

    Parameters:
    dataset_name (str): Specifies the type of problem for data generation,
                         e.g., "synthetic_linear_regression" or "synthetic_logistic_regression".
    dataset_params (dict): Dictionary containing parameters like n_dataset, true_theta,
                           param_dim, bias, cov_type, cov_const, diag, variance.
    device (str): The target device for the output tensors (true_theta, true_hessian).
    data_batch_size (int): Batch size for the iterable dataset to generate.
    rng_data (torch.Generator): Dedicated CPU generator for stochastic operations.
    data_gen_device (str): Device for the generation process itself.

    Returns:
    Tuple[RegressionIterableDataset, torch.Tensor, torch.Tensor | None]:
        A RegressionIterableDataset object,
        the true_theta tensor (on `device`),
        and the true_hessian tensor (on `device`, or None).
    """
    # Unpack parameters from dataset_params
    n_dataset: int = dataset_params["n_dataset"]
    true_theta_input: list | torch.Tensor | None = dataset_params.get("true_theta")
    param_dim_input: int | None = dataset_params.get("param_dim")
    bias: bool = dataset_params.get("bias", True)
    cov_type: str = dataset_params.get("cov_type", "identity")
    cov_const: float | None = dataset_params.get("cov_const")
    diag: bool = dataset_params.get("diag", False)
    variance: float = dataset_params.get("variance", 1.0)  # For linear regression noise

    if n_dataset <= 0:
        raise ValueError(f"Dataset size n_dataset must be positive, got {n_dataset}")
    if data_batch_size <= 0:
        raise ValueError(f"Data batch size must be positive, got {data_batch_size}")
    if not isinstance(cov_type, str):
        raise TypeError(f"cov_type must be a string, but got {type(cov_type)}")

    feature_dim: int
    actual_param_dim: int

    # Determine true_theta and its dimension, ensuring it's on the target `device`
    if true_theta_input is None:
        if param_dim_input is None:
            raise ValueError("Either 'true_theta' or 'param_dim' must be provided in dataset_params.")
        actual_param_dim = param_dim_input
        if actual_param_dim < 1 or (bias and actual_param_dim < 2):
            raise ValueError(f"param_dim must be at least 1 (bias=False) or 2 (bias=True), got {actual_param_dim}")
        feature_dim = actual_param_dim - 1 if bias else actual_param_dim
        true_theta = torch.randn(actual_param_dim, dtype=torch.float32, generator=rng_data).to(device)
    else:
        if isinstance(true_theta_input, list):
            true_theta = torch.tensor(true_theta_input, dtype=torch.float32).to(device)
        elif isinstance(true_theta_input, torch.Tensor):
            true_theta = true_theta_input.to(dtype=torch.float32, device=device)
        else:
            raise TypeError("true_theta must be a list, torch.Tensor, or None")

        actual_param_dim = len(true_theta)
        if actual_param_dim < 1:
            raise ValueError("Provided true_theta cannot be empty.")
        if bias and actual_param_dim < 2:
            raise ValueError(f"Provided true_theta length {actual_param_dim} invalid with bias=True.")

        if param_dim_input is not None and param_dim_input != actual_param_dim:
            print(
                f"Warning: param_dim ({param_dim_input}) and len(true_theta) ({actual_param_dim}) mismatch. Using len(true_theta)."
            )
        feature_dim = actual_param_dim - 1 if bias else actual_param_dim

    # Covariance matrix is used by the generator, so it must be on the generator's device
    true_feature_covariance_matrix = generate_covariance_matrix(
        feature_dim,
        cov_type=cov_type,
        cov_const=cov_const,
        diag=diag,
        device=data_gen_device,
        generator=rng_data,
    )

    dataset = RegressionIterableDataset(
        n_total_samples=n_dataset,
        true_theta=true_theta,
        theta_dim=actual_param_dim,
        bias=bias,
        cov_matrix=true_feature_covariance_matrix,
        data_batch_size=data_batch_size,
        variance=variance,
        dataset_name=dataset_name,
        rng_data=rng_data,
        data_gen_device=data_gen_device,
    )

    true_hessian: Optional[torch.Tensor] = None

    if dataset_name == "synthetic_linear_regression":
        cov_matrix_for_hessian = true_feature_covariance_matrix.to(device)

        if bias:
            if feature_dim > 0:
                true_hessian = torch.zeros((actual_param_dim, actual_param_dim), device=device, dtype=torch.float32)
                true_hessian[:-1, :-1] = cov_matrix_for_hessian
                true_hessian[-1, -1] = 1.0
            else:
                true_hessian = torch.tensor([[1.0]], device=device, dtype=torch.float32)
        else:
            true_hessian = cov_matrix_for_hessian

    return dataset, true_theta, true_hessian


def load_covtype_dataset(test_size: float, random_state: int, device: str) -> Tuple[MyDataset, MyDataset, int]:
    """
    Load the covtype dataset from sklearn, preprocess, and split it.

    Args:
    test_size (float): Proportion of the dataset for the test split.
    random_state (int): Random state for train_test_split.
    device (str): Device to load tensors onto.

    Returns:
    Tuple[MyDataset, MyDataset, int, str]:
        - Training dataset (MyDataset instance).
        - Testing dataset (MyDataset instance).
        - Parameter dimension (number of features).
    """
    name = "covtype"
    print(f"Loading {name} dataset...")

    # Fetch the dataset from sklearn
    covtype_data = fetch_covtype()
    X_np, y_np = covtype_data.data, covtype_data.target

    # Convert to binary classification: class 1 vs. all others (0)
    # In covtype, classes are 1-7. We map class 1 to 1, and others to 0.
    y_binary_np = np.where(y_np == 1, 1, 0)

    # Split the data
    X_train_np, X_test_np, Y_train_binary_np, Y_test_binary_np = train_test_split(
        X_np, y_binary_np, test_size=test_size, random_state=random_state, stratify=y_binary_np
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    Y_train_binary = torch.tensor(Y_train_binary_np, dtype=torch.float32, device=device).squeeze()
    X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    Y_test_binary = torch.tensor(Y_test_binary_np, dtype=torch.float32, device=device).squeeze()

    param_dim_data = X_train.shape[1]  # Number of features

    print(f"Finished loading and processing {name} dataset.")
    print(f"  Training X shape: {X_train.shape}, Training Y shape: {Y_train_binary.shape}")
    print(f"  Testing X shape: {X_test.shape}, Testing Y shape: {Y_test_binary.shape}")
    print(f"  Number of features (param_dim from data): {param_dim_data}")

    return (
        MyDataset(X_train, Y_train_binary),
        MyDataset(X_test, Y_test_binary),
        param_dim_data,
    )


def load_dataset_from_source(
    dataset_name: str, device: str, test_size: float, random_state: int = 0
) -> Tuple[MyDataset, MyDataset, int, int, int]:
    """
    Loads a specified dataset. For now, supports 'covtype'.
    Returns train/test datasets, param_dim, n_train, and n_test.
    """
    if dataset_name.lower() == "covtype":
        train_dataset, test_dataset, param_dim_data = load_covtype_dataset(
            test_size=test_size, random_state=random_state, device=device
        )
        return train_dataset, test_dataset, param_dim_data, train_dataset.n_samples, test_dataset.n_samples
    else:
        raise ValueError(f"Unknown dataset_name for loading: {dataset_name}")
