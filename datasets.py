import torch
import torch.distributions as dist
from torch.utils.data import Dataset, IterableDataset
from typing import Generator, Tuple, Optional
import math


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


class LinearRegressionIterableDataset(IterableDataset):
    def __init__(
        self,
        n_total_samples: int,
        true_theta: torch.Tensor,
        theta_dim: int,
        bias: bool,
        cov_matrix: torch.Tensor,
        device: str,
        data_batch_size: int,
        variance: float = 1.0,
    ):
        super().__init__()
        self.n_total_samples = n_total_samples
        self.true_theta = true_theta.to(device=device, dtype=torch.float32)
        self.bias = bias
        self.theta_dim = theta_dim
        self.feature_dim = self.theta_dim - 1 if self.bias else self.theta_dim
        self.device = device
        self.data_batch_size = data_batch_size
        self.cov_matrix = cov_matrix
        self.variance = variance

        self.feature_dist = None
        if self.feature_dim > 0:
            mean = torch.zeros(self.feature_dim, device=self.device, dtype=torch.float32)
            self.feature_dist = dist.MultivariateNormal(mean, covariance_matrix=self.cov_matrix)

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        num_batches = math.ceil(self.n_total_samples / self.data_batch_size)
        samples_yielded_so_far = 0

        for _ in range(num_batches):
            current_actual_batch_size = min(self.data_batch_size, self.n_total_samples - samples_yielded_so_far)
            if current_actual_batch_size <= 0:
                break

            X_batch: torch.Tensor
            if self.feature_dim > 0:
                if self.feature_dist is not None:
                    X_batch = self.feature_dist.sample(sample_shape=torch.Size([current_actual_batch_size]))
                else:
                    raise RuntimeError("feature_dist was not initialized though feature_dim > 0")
            else:
                X_batch = torch.empty((current_actual_batch_size, 0), device=self.device, dtype=torch.float32)

            phi_batch: torch.Tensor
            if self.bias:
                ones_batch = torch.ones((current_actual_batch_size, 1), device=self.device, dtype=X_batch.dtype)
                phi_batch = torch.cat([ones_batch, X_batch], dim=1)
            else:
                phi_batch = X_batch

            Y_batch = torch.normal(mean=torch.matmul(phi_batch, self.true_theta), std=math.sqrt(self.variance))

            samples_yielded_so_far += current_actual_batch_size
            yield X_batch, Y_batch


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


def random_matrix(d: int, cov_const: float | None = None, device: str = "cpu") -> torch.Tensor:
    if d == 0:
        return torch.empty((0, 0), device=device, dtype=torch.float32)
    if cov_const is None:
        cov_const = 2.0  # Default exponent for 1/j^cov_const
    if cov_const <= 0:  # Exponent should probably be positive for decay
        raise ValueError("cov_const (exponent) must be positive for this eigenvalue distribution.")

    # Generate a random orthogonal basis Q
    A = torch.randn(d, d, device=device, dtype=torch.float32)
    # Q, R from QR decomposition; Q is orthogonal.
    # However, to ensure Q is from a uniform distribution over orthogonal matrices (Haar measure),
    # it's common to start with a matrix of i.i.d. Gaussian entries and then apply QR.
    # If A has i.i.d. standard Gaussian entries, then Q from QR(A) is Haar distributed.
    Q, _ = torch.linalg.qr(A)
    # Ensure Q has the correct dtype if qr changed it (though unlikely for float32 input)
    Q = Q.to(dtype=torch.float32)

    # Define the new eigenvalues as 1/j^cov_const
    j_values = torch.arange(1, d + 1, device=device, dtype=torch.float32)
    new_eigenvalues = 1.0 / (j_values**cov_const)

    # Sort eigenvalues (e.g., ascending) for consistent diagonal matrix construction
    # This makes the largest eigenvalue 1.0 (for j=1) and smallest 1/d^cov_const.
    new_eigenvalues, _ = torch.sort(new_eigenvalues)  # Ascending by default

    # Reconstruct the new matrix with the random orthogonal eigenvectors Q and new eigenvalues
    # New_Cov_Matrix = Q @ diag(Lambda_new) @ Q.T
    new_cov_matrix = Q @ torch.diag(new_eigenvalues) @ Q.T

    # print(f"Target eigenvalues for random_matrix: {new_eigenvalues}") # Optional debug print
    print(f"Eigenvalues of the new matrix: {torch.sort(torch.linalg.eigvalsh(new_cov_matrix)).values}")
    return new_cov_matrix


def generate_covariance_matrix(
    feature_dim: int, cov_type: str, cov_const: float | None = None, diag: bool = False, device: str = "cpu"
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
            return random_matrix(feature_dim, cov_const=cov_const, device=device)
    except Exception as e:
        raise RuntimeError(f"Error generating covariance matrix of type '{cov_type}': {str(e)}")
    raise ValueError(f"Unhandled covariance matrix type: {cov_type}")


def generate_linear_regression(
    n_dataset: int,
    true_theta: list | torch.Tensor | None = None,
    param_dim: int | None = None,
    bias: bool = True,
    cov_type: str = "identity",
    cov_const: float | None = None,
    diag: bool = False,
    device: str = "cpu",
    data_batch_size: int = 1,
) -> Tuple[LinearRegressionIterableDataset, torch.Tensor, torch.Tensor]:
    """
    Generate data from a linear regression model using PyTorch, returning an iterable dataset.

    Parameters:
    n_dataset (int): Number of samples.
    true_theta (list | torch.Tensor | None, optional): True parameter vector.
    param_dim (int | None, optional): The dimension of the parameter vector `true_theta`.
    bias (bool): Whether to include a bias term.
    cov_type (str): The covariance matrix type for features.
    cov_const (float | None): The constant for the covariance matrix.
    diag (bool): Whether to modify the diagonal of the Toeplitz matrix for cov_type='toeplitz'.
    device (str): The device to create tensors on.
    data_batch_size (int): Batch size for the iterable dataset to generate.

    Returns:
    Tuple[LinearRegressionIterableDataset, torch.Tensor, torch.Tensor]:
        A LinearRegressionIterableDataset object,
        the true_theta tensor,
        and the true_hessian (covariance matrix of features X, not phi) tensor.
    """
    if n_dataset <= 0:
        raise ValueError(f"Dataset size n_dataset must be positive, got {n_dataset}")
    if data_batch_size <= 0:
        raise ValueError(f"Data batch size must be positive, got {data_batch_size}")
    if not isinstance(cov_type, str):
        raise TypeError(f"cov_type must be a string, but got {type(cov_type)}")

    feature_dim: int
    if true_theta is None:
        if param_dim is None:
            raise ValueError("Either 'true_theta' or 'param_dim' must be provided.")
        if param_dim < 1 or (bias and param_dim < 2):
            raise ValueError(f"param_dim must be at least 1 (bias=False) or 2 (bias=True), got {param_dim}")
        feature_dim = param_dim - 1 if bias else param_dim
        true_theta = torch.randn(param_dim, device=device, dtype=torch.float32)
    else:
        if isinstance(true_theta, list):
            true_theta = torch.tensor(true_theta, device=device, dtype=torch.float32)
        elif not isinstance(true_theta, torch.Tensor):
            raise TypeError("true_theta must be a list, torch.Tensor, or None")
        true_theta = true_theta.to(device=device, dtype=torch.float32)

        param_dim_from_theta = len(true_theta)
        if param_dim_from_theta < 1:
            raise ValueError("Provided true_theta cannot be empty.")
        if bias and param_dim_from_theta < 2:
            raise ValueError(f"Provided true_theta length {param_dim_from_theta} invalid with bias=True.")

        if param_dim is not None and param_dim != param_dim_from_theta:
            print(
                f"Warning: param_dim ({param_dim}) and len(true_theta) ({param_dim_from_theta}) mismatch. Using len(true_theta)."
            )
        param_dim = param_dim_from_theta
        feature_dim = param_dim - 1 if bias else param_dim

    true_feature_covariance_matrix = generate_covariance_matrix(
        feature_dim, cov_type=cov_type, cov_const=cov_const, diag=diag, device=device
    )

    # Construct the true Hessian for phi (features with bias if applicable)
    if bias:
        if feature_dim > 0:
            true_hessian = torch.zeros((param_dim, param_dim), device=device, dtype=torch.float32)
            true_hessian[0, 0] = 1.0
            # E[X] is assumed to be 0, so off-diagonal blocks involving bias are 0
            true_hessian[1:, 1:] = true_feature_covariance_matrix
        else:  # Only bias term, param_dim is 1
            true_hessian = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    else:  # No bias, phi = X
        true_hessian = true_feature_covariance_matrix

    dataset = LinearRegressionIterableDataset(
        n_total_samples=n_dataset,
        true_theta=true_theta,
        theta_dim=param_dim,
        bias=bias,
        cov_matrix=true_feature_covariance_matrix,
        device=device,
        data_batch_size=data_batch_size,
    )

    return dataset, true_theta, true_hessian
