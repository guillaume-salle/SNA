import torch
import torch.distributions as dist
from torch.utils.data import Dataset, IterableDataset
from typing import Generator, Tuple, Optional
import math
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import openml
import pandas as pd
from torchvision import datasets as tv_datasets
from torchvision.transforms import ToTensor
import time


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


def toeplitz_matrix(d: int, cov_const: float | None = None, device: str = "cpu") -> torch.Tensor:
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
        cov_const = 100.0  # Default: results in eigenvalues [1, ..., 1/100]

    if not isinstance(cov_const, (float, int)) or cov_const <= 0:
        raise ValueError(
            f"'cov_const' (value defining eigenvalue range [1, 1/cov_const]) must be positive, got {cov_const}."
        )

    A = torch.randn(d, d, device=device, dtype=torch.float32, generator=generator)
    Q, _ = torch.linalg.qr(A)
    Q = Q.to(dtype=torch.float32)

    # Generate d logarithmically spaced eigenvalues between 1.0 and 1.0/cov_const
    log_eigenvalue_start = torch.tensor(0.0, device=device, dtype=torch.float32)  # log(1.0)

    # Ensure cov_const is a tensor for torch.log
    cov_const_tensor = torch.tensor(cov_const, device=device, dtype=torch.float32)
    log_eigenvalue_end = -torch.log(cov_const_tensor)  # log(1.0 / cov_const)

    if d == 1:
        # For d=1, torch.linspace(start, end, 1) yields 'start'.
        # So, the single eigenvalue will be exp(log_eigenvalue_start) = 1.0.
        log_spaced_values = log_eigenvalue_start.unsqueeze(0)
    else:
        log_spaced_values = torch.linspace(
            log_eigenvalue_start, log_eigenvalue_end, d, device=device, dtype=torch.float32
        )

    new_eigenvalues = torch.exp(log_spaced_values)

    # Sort eigenvalues (e.g., ascending) for consistent behavior,
    # though linspace followed by exp might already be ordered depending on cov_const.
    new_eigenvalues, _ = torch.sort(new_eigenvalues)

    new_cov_matrix = Q @ torch.diag(new_eigenvalues) @ Q.T

    # print(f"Target eigenvalues for random_matrix: {new_eigenvalues}")
    eigenvalues_check = torch.linalg.eigvalsh(new_cov_matrix)
    min_eig = torch.min(eigenvalues_check)
    max_eig = torch.max(eigenvalues_check)
    # The print statement below helps verify the eigenvalue range.
    # Note: Due to numerical precision, actual min/max might slightly differ from 1.0 and 1.0/cov_const.
    print(f"    Min eigenvalue of the generated random matrix: {min_eig:.4e}, Max eigenvalue: {max_eig:.4e}")
    return new_cov_matrix


def generate_covariance_matrix(
    feature_dim: int,
    cov_type: str,
    cov_const: float | None = None,
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
            return toeplitz_matrix(feature_dim, cov_const=cov_const, device=device)
        elif cov_type == "hard":
            return hard_matrix(feature_dim, cov_const=cov_const, device=device)
        elif cov_type == "random":
            return random_matrix(feature_dim, cov_const=cov_const, device=device, generator=generator)
    except Exception as e:
        raise RuntimeError(f"Error generating covariance matrix of type '{cov_type}': {str(e)}") from e
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


def _split_and_convert_to_torch(
    X_np: np.ndarray,
    y_np: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
    dataset_name: str,
) -> Tuple[MyDataset, MyDataset, MyDataset, int]:
    """
    Splits numpy arrays into train, validation, and test sets and converts them to MyDataset objects.
    This function contains the common logic for data splitting and tensor conversion.
    """
    print(f"Total dataset size: {len(X_np)} samples.")
    # Split data into training+validation and test sets
    stratify_np = y_np if np.unique(y_np).size > 1 and np.min(np.bincount(y_np.astype(int))) > 1 else None
    X_train_temp_np, X_test_np, y_train_temp_np, y_test_np = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state, stratify=stratify_np
    )

    # Split training set into actual training and validation sets
    if val_size > 0 and len(X_train_temp_np) > 0:
        val_test_proportion = val_size / (1.0 - test_size)
        stratify_train_temp = (
            y_train_temp_np
            if np.unique(y_train_temp_np).size > 1 and np.min(np.bincount(y_train_temp_np.astype(int))) > 1
            else None
        )
        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_train_temp_np,
            y_train_temp_np,
            test_size=val_test_proportion,
            random_state=random_state,
            stratify=stratify_train_temp,
        )
    else:
        X_train_np, y_train_np = X_train_temp_np, y_train_temp_np
        X_val_np = (
            np.array([], dtype=np.float32).reshape(0, X_train_np.shape[1])
            if X_train_np.shape[0] > 0
            else np.array([], dtype=np.float32).reshape(0, X_np.shape[1])
        )
        y_val_np = np.array([], dtype=np.float32)

    # Convert to PyTorch tensors on CPU
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device="cpu")
    Y_train = torch.tensor(y_train_np, dtype=torch.float32, device="cpu").squeeze()
    X_val = torch.tensor(X_val_np, dtype=torch.float32, device="cpu")
    Y_val = torch.tensor(y_val_np, dtype=torch.float32, device="cpu").squeeze()
    X_test = torch.tensor(X_test_np, dtype=torch.float32, device="cpu")
    Y_test = torch.tensor(y_test_np, dtype=torch.float32, device="cpu").squeeze()

    number_features = X_train.shape[1] if X_train.shape[0] > 0 else X_np.shape[1]
    print(f"Finished splitting and converting {dataset_name} dataset.")
    print(f"  Training X shape: {X_train.shape}, Training Y shape: {Y_train.shape}")
    print(f"  Validation X shape: {X_val.shape}, Validation Y shape: {Y_val.shape}")
    print(f"  Testing X shape: {X_test.shape}, Testing Y shape: {Y_test.shape}")
    print(f"  Number of features from data: {number_features}")

    return (
        MyDataset(X_train, Y_train),
        MyDataset(X_val, Y_val),
        MyDataset(X_test, Y_test),
        number_features,
    )


def load_openml_dataset(
    dataset_name: str, test_size: float, val_size: float, random_state: int
) -> Tuple[MyDataset, MyDataset, MyDataset, int]:
    """
    Fetch a dataset from OpenML using scikit-learn's fetch_openml.
    It uses as_frame='auto' to handle both dense and sparse datasets.
    """
    print(f"Loading and processing {dataset_name} dataset via sklearn.fetch_openml...")

    version_to_fetch = "active"
    if dataset_name == "connect-4":
        print("  Note: Using version 2 of connect-4 dataset to avoid issues with version 1.")
        version_to_fetch = 2
    elif dataset_name == "PhishingWebsites":
        print("  Note: Using version 2 of PhishingWebsites dataset.")
        version_to_fetch = 1
    elif dataset_name == "adult":
        print("  Note: Using version 2 of adult dataset.")
        version_to_fetch = 2
    elif dataset_name == "mushroom":
        print("  Note: Using version 1 of mushroom dataset.")
        version_to_fetch = 1

    start_time = time.time()
    bunch = fetch_openml(name=dataset_name, as_frame="auto", parser="auto", version=version_to_fetch)
    duration = time.time() - start_time
    print(f"  Fetching data took {duration:.2f} seconds.")

    X, y = bunch.data, bunch.target

    X_np: np.ndarray
    y_np: np.ndarray

    if isinstance(X, pd.DataFrame):
        print("  Info: Dataset loaded as a pandas DataFrame.")
        # Identify categorical features to one-hot encode them
        categorical_cols = X.select_dtypes(include=["category", "object"]).columns
        if not categorical_cols.empty:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # Convert any remaining non-numeric columns (like bools) to int
        for col in X.columns:
            if X[col].dtype not in [np.float32, np.float64, np.int32, np.int64]:
                try:
                    X[col] = X[col].astype(float).astype(np.float32)
                except ValueError:
                    print(f"  Warning: Could not convert column '{col}' to float.")
        X_np = X.to_numpy(dtype=np.float32)
    else:
        print("  Info: Dataset loaded as a numpy/scipy array (likely sparse).")
        # Handle sparse matrix case
        if hasattr(X, "toarray"):
            X_np = X.toarray()
        else:
            X_np = X
        X_np = X_np.astype(np.float32)

    # Process target variable y to ensure it's binary {0, 1}.
    y_series = y if isinstance(y, pd.Series) else pd.Series(y)
    y_categorical = y_series.astype("category")
    num_classes = len(y_categorical.cat.categories)

    print(f"  Info: Original unique labels found: {y_categorical.cat.categories.to_list()}")

    if num_classes > 2:
        print(
            f"  Info: Dataset '{dataset_name}' has {num_classes} classes. "
            "Converting to binary by mapping the most frequent class to 1 and others to 0."
        )
        major_class_label = y_categorical.value_counts().idxmax()
        print(f"  Info: Most frequent class is '{major_class_label}'.")
        y_np = (y_series == major_class_label).to_numpy(dtype=np.float32)
        print(f"  Info: New unique labels: {np.unique(y_np).tolist()}")
    elif num_classes == 2:
        # cat.codes will map the two classes to {0, 1}
        y_np = y_categorical.cat.codes.to_numpy(dtype=np.float32)
        print("  Info: Labels successfully mapped to {0, 1}.")
    else:  # num_classes < 2
        raise ValueError(
            f"Dataset '{dataset_name}' has fewer than 2 classes, which is not supported. "
            f"Found {num_classes} classes: {y_categorical.cat.categories.to_list()}"
        )

    return _split_and_convert_to_torch(X_np, y_np, test_size, val_size, random_state, dataset_name)


def load_mnist_dataset(
    test_size: float, val_size: float, random_state: int
) -> Tuple[MyDataset, MyDataset, MyDataset, int]:
    """
    Load and process the MNIST dataset for binary classification of even vs. odd digits.
    Even digits {0, 2, 4, 6, 8} are mapped to class 0.
    Odd digits {1, 3, 5, 7, 9} are mapped to class 1.
    """
    print("Loading and processing MNIST dataset (Even vs. Odd)...")
    # Load raw data
    train_data = tv_datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = tv_datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

    # Combine data
    X_combined = torch.cat((train_data.data, test_data.data), dim=0).float()
    y_combined = torch.cat((train_data.targets, test_data.targets), dim=0)

    # Remap targets to binary: 0 for even, 1 for odd
    y_binary = y_combined % 2

    # Flatten images and convert to numpy
    X_np = X_combined.reshape(X_combined.shape[0], -1).numpy()
    y_np = y_binary.numpy()

    return _split_and_convert_to_torch(X_np, y_np, test_size, val_size, random_state, "MNIST")


def load_covtype_dataset_sklearn(
    test_size: float, val_size: float, random_state: int
) -> Tuple[MyDataset, MyDataset, MyDataset, int]:
    """
    Load and process the covtype dataset from sklearn.
    The task is converted to a binary classification problem: class 2 vs. all others.
    """
    print("Loading and processing covtype dataset from sklearn...")
    X, y = fetch_covtype(return_X_y=True)

    # Convert to binary classification: class 2 (Lodgepole Pine) vs all others.
    y_binary = (y == 2).astype(np.float32)

    X_np = X.astype(np.float32)
    y_np = y_binary

    return _split_and_convert_to_torch(X_np, y_np, test_size, val_size, random_state, "covtype_sklearn")


def load_dataset_from_source(
    dataset_name: str, test_size: float, val_size: float, random_state: int = 0, **kwargs
) -> dict:
    """
    Loads a specified dataset.
    Returns a dictionary containing train/val/test datasets, param_dim, and counts.
    """
    dataset_name_lower = dataset_name.lower()
    train_dataset, val_dataset, test_dataset, number_features = (None, None, None, None)

    # Map user-friendly names to the names required by sklearn.fetch_openml
    openml_name_map = {
        "covtype": "covertype",
        "mushrooms": "mushroom",
        "adult": "adult",
        "phishing": "PhishingWebsites",
        "connect-4": "connect-4",
        "gisette": "gisette",
    }

    if dataset_name_lower in openml_name_map:
        fetch_name = openml_name_map[dataset_name_lower]
        train_dataset, val_dataset, test_dataset, number_features = load_openml_dataset(
            dataset_name=fetch_name,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )
    elif dataset_name_lower == "mnist":
        train_dataset, val_dataset, test_dataset, number_features = load_mnist_dataset(
            test_size=test_size, val_size=val_size, random_state=random_state
        )
    elif dataset_name_lower == "covtype_sklearn":
        train_dataset, val_dataset, test_dataset, number_features = load_covtype_dataset_sklearn(
            test_size=test_size, val_size=val_size, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown dataset_name for loading: {dataset_name}")

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "number_features": number_features,
        "n_train": train_dataset.n_samples,
        "n_test": test_dataset.n_samples,
        "n_val": val_dataset.n_samples,
    }
