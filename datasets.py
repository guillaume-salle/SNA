import torch
from torch.utils.data import Dataset
from typing import Generator, Tuple, Optional


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


def toeplitz_matrix(d: int, cov_const: float | None = None, diag: bool = False, device: str = "cpu") -> torch.Tensor:
    """
    Generate a Toeplitz matrix of size d with a modified diagonal using PyTorch.

    Parameters:
    d (int): Size of the matrix.
    cov_const (float): Value to be taken to power of the absolute difference between the indices. Default is 1/d.
    diag (bool): Whether the diagonal should be modified to 1+i. If False, diagonal is 1.
    device (str): The device to create the tensor on ('cpu' or 'cuda').

    Returns:
    torch.Tensor: Toeplitz matrix.
    """
    if cov_const is None:
        cov_const = 1.0 / d
    indices = torch.arange(d, device=device)
    # Calculate absolute differences using broadcasting
    abs_diff = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))
    # Calculate matrix values: const ^ |i-j|
    # Ensure const is a tensor for pow operation
    const_tensor = torch.tensor(cov_const, device=device, dtype=torch.float32)
    matrix = torch.pow(const_tensor, abs_diff.float())

    if diag:
        # Set diagonal elements to 1 + i
        diag_indices = torch.arange(d, device=device)
        diag_values = (1.0 + diag_indices).to(matrix.dtype)
        # Efficiently set the diagonal
        matrix.diagonal().copy_(diag_values)

    return matrix


def hard_matrix(d: int, cov_const: float | None = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate a matrix difficult for the mask technique of USNA.
    diagonal blocs of size 2x2 with values 1/const, 1, 1, const + 1.
    """
    if d % 2 != 0:
        raise ValueError("d must be even for hard matrix")
    if cov_const is None:
        cov_const = 100.0
    matrix = torch.zeros(d, d, device=device, dtype=torch.float32)
    block = torch.tensor([[1 / cov_const, 1], [1, cov_const + 1]], device=device, dtype=torch.float32)
    for i in range(d // 2):
        matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
    return matrix


def generate_covariance_matrix(
    feature_dim: int, cov_type: str, cov_const: float | None = None, diag: bool = False, device: str = "cpu"
) -> torch.Tensor:
    if cov_type == "identity":
        return torch.eye(feature_dim, device=device, dtype=torch.float32)
    elif cov_type == "toeplitz":
        return toeplitz_matrix(feature_dim, cov_const=cov_const, diag=diag, device=device)
    elif cov_type == "hard":
        return hard_matrix(feature_dim, cov_const=cov_const, device=device)
    else:
        raise ValueError(f"Invalid covariance matrix type: {cov_type}")


def generate_linear_regression(
    n_dataset: int,
    true_theta: list | torch.Tensor | None = None,
    param_dim: int | None = None,
    bias: bool = True,
    cov_type: torch.Tensor | str = "identity",
    cov_const: float | None = None,
    diag: bool = False,
    device: str = "cpu",
) -> Tuple[MyDataset, torch.Tensor]:
    """
    Generate data from a linear regression model using PyTorch.

    Parameters:
    n_dataset (int): Number of samples.
    true_theta (list | torch.Tensor | None, optional): True parameter vector.
                                                      If None, `param_dim` must be provided to generate a random true_theta.
                                                      The length of this vector defines the parameter dimension.  Defaults to None.
    param_dim (int | None, optional): The dimension of the parameter vector `true_theta` (including bias if `bias=True`).
                                      Required if `true_theta` is None. If `true_theta` is provided, this is optional
                                      and used for consistency check. Defaults to None.
    bias (bool): Whether to include a bias term (intercept) as the first component of `true_theta`
                 and a corresponding column of ones in the features. Defaults to True.
    cov_matrix (torch.Tensor | str): The covariance matrix to use for the features. Defaults to "identity".
    cov_matrix_const (float | None): The constant for the covariance matrix. Defaults to None.
    diag (bool): Whether to modify the diagonal of the Toeplitz matrix. Defaults to False.
    device (str): The device to create tensors on ('cpu' or 'cuda'). Defaults to "cpu".

    Returns:
    Tuple[MyDataset, torch.Tensor]: A MyDataset object containing generated torch.Tensors (X, Y)
                                    and the true_theta tensor (shape: [param_dim]).

    Raises:
    ValueError: If both `true_theta` and `param_dim` are None, or if they are inconsistent,
                or if `param_dim` is invalid (e.g., < 1, or < 2 if bias=True).
    TypeError: If `true_theta` is not a list, Tensor, or None.

    Note: Assumes MyDataset class (defined elsewhere) is adapted to handle torch.Tensors.
    """
    feature_dim: int

    # --- Determine true_theta and feature_dim ---
    if true_theta is None:
        if param_dim is None:
            raise ValueError("Either 'true_theta' or 'param_dim' must be provided to generate linear regression data.")
        if param_dim < 1 or (bias and param_dim < 2):
            raise ValueError(f"param_dim must be at least 1 (bias=False) or 2 (bias=True), got {param_dim}")
        feature_dim = param_dim - 1 if bias else param_dim

        # Generate true_theta from standard normal distribution
        true_theta = torch.randn(param_dim, device=device, dtype=torch.float32)
        print(
            f"   Generated random true_theta with param_dim={param_dim}, bias={bias} => feature_dim={feature_dim}. Shape={true_theta.shape}"
        )

    else:
        # Convert true_theta to tensor if it's a list and ensure device and dtype
        if isinstance(true_theta, list):
            true_theta = torch.tensor(true_theta, device=device, dtype=torch.float32)
        elif not isinstance(true_theta, torch.Tensor):
            raise TypeError("true_theta must be a list, torch.Tensor, or None")
        true_theta = true_theta.to(device=device, dtype=torch.float32)  # Ensure correct device and dtype

        # Determine dimensions from the provided true_theta
        param_dim = len(true_theta)
        if param_dim < 1:
            raise ValueError("Provided true_theta cannot be empty.")
        if bias and param_dim < 2:
            raise ValueError(f"Provided true_theta has length {param_dim}, but bias=True requires length at least 2.")

        feature_dim = param_dim - 1 if bias else param_dim

        # If true_theta is provided, its length defines the dimensions. Ignore param_dim if it was passed.
        print(
            f"   Using provided true_theta. Length={param_dim}, bias={bias} => feature_dim={feature_dim}. Shape={true_theta.shape}"
        )

    # --- Generate Data using feature_dim and true_theta ---

    # Ensure true_theta is float32 for subsequent operations (redundant due to checks above, but safe)
    if true_theta.dtype != torch.float32:
        true_theta = true_theta.to(dtype=torch.float32)

    # Generate features X (using feature_dim and covariance matrix)
    if feature_dim > 0:
        if cov_type == "identity":
            loc = torch.zeros(feature_dim, device=device, dtype=torch.float32)
            scale = torch.ones(feature_dim, device=device, dtype=torch.float32)
            base_dist = dist.Normal(loc, scale)
            mvn = dist.Independent(base_dist, reinterpreted_batch_ndims=1)
        else:
            mean = torch.zeros(feature_dim, device=device, dtype=torch.float32)
            covariance_matrix = generate_covariance_matrix(
                feature_dim, cov_type=cov_type, cov_const=cov_const, diag=diag, device=device
            )
            print(f"covariance_matrix: {covariance_matrix}")
            mvn = dist.MultivariateNormal(mean, covariance_matrix=covariance_matrix)

        X = mvn.sample(sample_shape=torch.Size([n_dataset]))  # Shape (n, feature_dim)
    else:  # Handle case where there are no features (only bias)
        X = torch.empty((n_dataset, 0), device=device, dtype=torch.float32)  # Shape (n, 0)

    # Add bias term if needed to create phi
    if bias:
        ones_col = torch.ones((n_dataset, 1), device=device, dtype=X.dtype)
        phi = torch.cat([ones_col, X], dim=1)  # Shape (n, 1 + feature_dim) = (n, actual_param_dim)
    else:
        phi = X  # Shape (n, feature_dim) = (n, actual_param_dim)

    # Generate labels Y = phi @ true_theta + noise
    noise = torch.randn(n_dataset, device=device, dtype=X.dtype)
    Y = torch.matmul(phi, true_theta) + noise  # Shape (n,)

    return MyDataset(X=X, Y=Y), true_theta
