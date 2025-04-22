import torch
from torch.distributions import MultivariateNormal
from .BaseDataGenerator import BaseDataGenerator


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
        if not (-1.0 / (dim - 1) < rho < 1.0):
            raise ValueError(
                f"Equicorrelation parameter rho={rho} must be in the range"
                f" (-1/({dim-1})={-1.0 / (dim - 1):.4f}, 1.0) for positive definiteness."
            )
        Sigma = torch.full((dim, dim), rho, dtype=torch.float32, device=device)
        Sigma.fill_diagonal_(1.0)
    else:
        raise ValueError(
            f"Unknown covariance_type: '{cov_type}'. Choose from 'identity', 'diagonal', 'ar1', 'equicorrelation'."
        )

    print(f"Successfully created Sigma matrix for type '{cov_type}'.")
    return Sigma


class LinearModelGenerator(BaseDataGenerator):
    """Generates data for y = X*theta + noise, allows custom X covariance."""

    def __init__(
        self, seed, device, dim, noise_std=0.1, covariance_type="identity", covariance_param=None
    ):
        super().__init__(seed, device)
        self.dim = dim
        self.noise_std = noise_std
        self.covariance_type = covariance_type
        self.covariance_param = covariance_param

        self._setup_x_distribution()
        self.theta_true = torch.randn(self.dim, 1, dtype=torch.float32, device=self.device)
        print(
            f"LinearModelGenerator initialized with dim={dim}, noise_std={noise_std}, cov_type='{covariance_type}'"
        )

    def _setup_x_distribution(self):
        """Sets up the distribution specificially for generating X in this model."""
        # This logic might be duplicated if other models need the same X,
        # consider helper functions/classes if that happens frequently.
        print("Setting up X distribution for LinearModel...")
        mean_vector = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        try:
            Sigma = _create_covariance_matrix(
                self.dim, self.covariance_type, self.covariance_param, self.device
            )
            self.mvn_distribution_x = MultivariateNormal(loc=mean_vector, covariance_matrix=Sigma)
            print(f"Created MultivariateNormal for X.")
        except (torch.linalg.LinAlgError, ValueError) as e:
            print(f"\n!!! Failed to create distribution for X. Error: {e}\n")
            raise

    def generate_chunk(self, chunk_size):
        X_chunk = self.mvn_distribution_x.sample((chunk_size,))
        noise = torch.randn(chunk_size, 1, dtype=torch.float32, device=self.device) * self.noise_std
        y_chunk = torch.matmul(X_chunk, self.theta_true) + noise
        return X_chunk, y_chunk

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.update(
            {
                "model_type": "linear_regression",
                "dim": self.dim,
                "noise_std": self.noise_std,
                "covariance_type": self.covariance_type,
                "covariance_param": (
                    str(self.covariance_param) if self.covariance_param is not None else "None"
                ),
            }
        )
        return metadata

    def get_true_parameters(self):
        return {"theta_true": self.theta_true}
