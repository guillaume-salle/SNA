from abc import ABC, abstractmethod
import torch
from typing import Tuple


class BaseObjectiveFunction(ABC):
    """
    Abstract base class for different objective functions using PyTorch.
    """

    @abstractmethod
    def __call__(self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """Compute the objective function value."""
        raise NotImplementedError

    @abstractmethod
    def get_param_dim(self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> int:
        """Return the dimension of the parameter vector."""
        raise NotImplementedError

    @abstractmethod
    def grad(self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the objective function."""
        raise NotImplementedError

    @abstractmethod
    def hessian(self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """Compute the Hessian of the objective function."""
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian(
        self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both the gradient and the Hessian."""
        raise NotImplementedError

    @abstractmethod
    def hessian_column(
        self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, z: int
    ) -> torch.Tensor:
        """Compute a specific column of the Hessian."""
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian_column(
        self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, z: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the gradient and a specific column of the Hessian."""
        raise NotImplementedError

    @abstractmethod
    def hessian_vector(
        self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, vector: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Hessian-vector product."""
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian_vector(
        self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the gradient and the Hessian-vector product."""
        raise NotImplementedError


class LinearRegression(BaseObjectiveFunction):
    """
    Linear Regression class using PyTorch.
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Linear"

    def _add_bias(self, X: torch.Tensor) -> torch.Tensor:
        """Helper function to add bias term (column of ones) to X."""
        if not self.bias:
            return X
        # Assume X is always batched (N, d)
        batch_size = X.size(0)
        ones = torch.ones((batch_size, 1), device=X.device, dtype=X.dtype)
        return torch.cat([ones, X], dim=1)

    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear regression loss (mean squared error), assumes batched input.
        Returns the average loss over the batch.
        """
        X, y = data
        phi = self._add_bias(X)  # phi now includes bias if self.bias is True
        Y_pred = torch.matmul(phi, param)

        loss = 0.5 * torch.mean((Y_pred - y) ** 2)  # Average loss over the batch
        return loss

    def get_param_dim(self, data: Tuple[torch.Tensor, torch.Tensor]) -> int:
        """
        Return the dimension of theta, works with a batch or a single data point.
        """
        X, _ = data
        feature_dim = X.shape[-1]
        return feature_dim + 1 if self.bias else feature_dim

    def grad(self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the linear regression loss, averaged over the batch. Assumes batched input.
        """
        X, y = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, d+1)

        Y_pred = torch.matmul(phi, param)  # Shape (batch_size,)
        error = Y_pred - y  # Shape (batch_size,)
        # Gradient: (phi.T @ error) / batch_size
        # phi.T shape: (d+1, batch_size), error shape: (batch_size,) -> result (d+1,)
        grad = torch.matmul(phi.T, error) / batch_size
        return grad

    def hessian(self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian of the linear regression loss, averaged over the batch. Assumes batched input.
        For linear regression, the Hessian is independent of the parameters `param`.
        H = (1/batch_size) * phi.T @ phi
        """
        X, _ = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, d+1)

        # phi.T shape: (d+1, batch_size), phi shape: (batch_size, d+1) -> result (d+1, d+1)
        hessian = torch.matmul(phi.T, phi) / batch_size
        return hessian

    def hessian_column(self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, col: int) -> torch.Tensor:
        """
        Compute a single column of the hessian, averaged over the batch. Assumes batched input.
        H_col = (1/batch_size) * phi.T @ phi[:, col]
        """
        X, _ = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, d+1)

        # phi.T shape: (d+1, batch_size), phi[:, col] shape: (batch_size,) -> result (d+1,)
        hessian_col = torch.matmul(phi.T, phi[:, col]) / batch_size
        return hessian_col

    def grad_and_hessian_column(
        self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, col: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and a single column of the hessian, averaged over the batch. Assumes batched input.
        """
        X, y = data
        batch_size = X.size(0)  # Assumes X has batch dimension
        phi = self._add_bias(X)  # Shape (batch_size, d+1)

        # Gradient calculation
        Y_pred = torch.matmul(phi, param)  # Shape (batch_size,)
        error = Y_pred - y  # Shape (batch_size,)
        grad = torch.matmul(phi.T, error) / batch_size  # Shape (d+1,)

        # Hessian column calculation
        hessian_col = torch.matmul(phi.T, phi[:, col]) / batch_size  # Shape (d+1,)

        return grad, hessian_col

    def grad_and_hessian(
        self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and the Hessian, averaged over the batch. Assumes batched input.
        """
        X, y = data
        batch_size = X.size(0)  # Assumes X has batch dimension
        phi = self._add_bias(X)  # Shape (batch_size, d+1)

        # Gradient calculation
        Y_pred = torch.matmul(phi, param)  # Shape (batch_size,)
        error = Y_pred - y  # Shape (batch_size,)
        grad = torch.matmul(phi.T, error) / batch_size  # Shape (d+1,)

        # Hessian calculation
        hessian = torch.matmul(phi.T, phi) / batch_size  # Shape (d+1, d+1)

        return grad, hessian

    def hessian_vector(
        self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Hessian-vector product (or a product with a matrix of k vectors) for linear regression.

        For linear regression with mean squared error loss:
        - The Hessian is H = (1/batch_size) * phi.T @ phi
        - The product is H @ V = (1/batch_size) * phi.T @ (phi @ V)

        Args:
            data: Tuple of (X, y) where X is the feature matrix and y is the target vector.
                  Note: y is not used for Hessian calculation in linear regression.
            param: Current parameter vector (unused in linear regression as Hessian is constant).
            vector (torch.Tensor): A 1D tensor of shape (param_dim,) or a 2D tensor of shape (param_dim, k)
                                   representing one or k vectors to multiply with the Hessian.

        Returns:
            torch.Tensor: The Hessian-vector product. Shape will be (param_dim, k).
        """
        X, _ = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, param_dim)

        if vector.ndim == 1:
            # Unsqueeze 1D vector to 2D (param_dim, 1) for consistent matrix multiplication
            vector_2d = vector.unsqueeze(1)
        elif vector.ndim == 2:
            vector_2d = vector
        else:
            raise ValueError(
                f"Unsupported vector shape: {vector.shape}. Expected 1D (param_dim,) or 2D (param_dim, k)."
            )

        # vector_2d is now (param_dim, k) where k can be 1
        # phi @ vector_2d results in (batch_size, k)
        phi_v = torch.matmul(phi, vector_2d)
        # phi.T @ phi_v results in (param_dim, k)
        hessian_vector_product = torch.matmul(phi.T, phi_v) / batch_size

        return hessian_vector_product

    def grad_and_hessian_vector(
        self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Compute phi ONCE

        # Gradient calculation (uses phi)
        Y_pred = torch.matmul(phi, param)
        error = Y_pred - y
        grad = torch.matmul(phi.T, error) / batch_size

        # Hessian-vector product calculation (uses phi)
        if vector.ndim == 1:
            vector_2d = vector.unsqueeze(1)
        elif vector.ndim == 2:
            vector_2d = vector
        else:
            raise ValueError(
                f"Unsupported vector shape: {vector.shape}. Expected 1D (param_dim,) or 2D (param_dim, k)."
            )

        # vector_2d is (param_dim, k)
        phi_v = torch.matmul(phi, vector_2d)  # phi @ V -> (batch_size, k)
        hess_vec_prod = torch.matmul(phi.T, phi_v) / batch_size  # phi.T @ (phi @ V) -> (param_dim, k)

        return grad, hess_vec_prod

    def sherman_morrison(
        self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, n_iter: int = None
    ) -> torch.Tensor:
        """
        Compute the Sherman-Morrison term (feature vector phi), works only for a batch size of 1. Assumes batched input.
        """
        X, _ = data
        # Ensure batch size is 1, assuming X already has batch dimension
        if X.size(0) != 1:
            raise ValueError("The Sherman-Morrison update is only possible for a batch size of 1")

        # Squeeze the batch dimension to get the 1D feature vector
        X = X.squeeze(0)  # Shape (d,)

        phi = self._add_bias(X)  # Shape (d+1,)
        return phi  # The Sherman-Morrison term is just the feature vector phi

    def grad_and_sherman_morrison(
        self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, n_iter: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and the Sherman-Morrison term, works only for a batch size of 1.
        Note: Gradient here is the *stochastic* gradient (not averaged).
        """
        X, y = data
        # Ensure batch size is 1
        if X.dim() > 1 and X.size(0) != 1:
            raise ValueError("The Sherman-Morrison update is only possible for a batch size of 1")

        # Squeeze batch dimension if present
        if X.dim() > 1:
            X = X.squeeze(0)  # Shape (d,)
            y = y.squeeze(0)  # Shape ()

        phi = self._add_bias(X)  # Shape (d+1,)

        # Stochastic Gradient calculation: (phi @ param - y) * phi
        Y_pred = torch.dot(phi, param)  # Scalar prediction
        error = Y_pred - y  # Scalar error
        stochastic_grad = error * phi  # Shape (d+1,)

        sherman_morrison_term = phi  # Shape (d+1,)

        return stochastic_grad, sherman_morrison_term
