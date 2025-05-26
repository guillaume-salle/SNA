from abc import ABC, abstractmethod
import torch
from typing import Tuple, Union


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
    def hessian(
        self, data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, return_grad: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the Hessian of the objective function. Optionally returns the gradient as well."""
        raise NotImplementedError

    @abstractmethod
    def hessian_column(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        param: torch.Tensor,
        columns: torch.Tensor,
        return_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute specific column(s) of the Hessian. Optionally returns the gradient as well. Assumes columns is a 1D LongTensor."""
        raise NotImplementedError

    @abstractmethod
    def hessian_vector(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        param: torch.Tensor,
        vector: torch.Tensor,
        return_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the Hessian-vector product. Optionally returns the gradient as well."""
        raise NotImplementedError

    def _add_bias(self, X: torch.Tensor) -> torch.Tensor:
        """Helper function to add bias term (column of ones) to X."""
        if not self.bias:
            return X
        # Assume X is always batched (N, d)
        batch_size = X.size(0)
        ones = torch.ones((batch_size, 1), device=X.device, dtype=X.dtype)
        return torch.cat([X, ones], dim=1)


class LinearRegression(BaseObjectiveFunction):
    """
    Linear Regression class using PyTorch.
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Linear Regression"

    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear regression loss (mean squared error), assumes batched input.
        Returns the average loss over the batch.
        """
        X, y = data
        phi = self._add_bias(X)
        Y_pred = torch.matmul(phi, param)

        loss = 0.5 * torch.mean((Y_pred - y) ** 2)
        return loss

    def get_param_dim(self, data: Tuple[torch.Tensor, torch.Tensor]) -> int:
        """
        Return the dimension of the parameter vector.
        """
        X, _ = data
        feature_dim = X.shape[-1]
        return feature_dim + 1 if self.bias else feature_dim

    def _grad_internal(self, phi: torch.Tensor, y: torch.Tensor, param: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Helper to compute gradient from phi, y, param, and batch_size."""
        Y_pred = torch.matmul(phi, param)  # Shape (batch_size,)
        error = Y_pred - y  # Shape (batch_size,)
        # Gradient: (phi.T @ error) / batch_size
        # phi.T shape: (d+1, batch_size), error shape: (batch_size,) -> result (d+1,)
        grad = torch.matmul(phi.T, error) / batch_size
        return grad

    def grad(self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the linear regression loss, averaged over the batch. Assumes batched input.
        """
        X, y = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, d+1)

        grad = self._grad_internal(phi, y, param, batch_size)
        return grad

    def hessian(
        self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, return_grad: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the Hessian of the linear regression loss, averaged over the batch. Assumes batched input.
        For linear regression, the Hessian is independent of the parameters `param`.
        H = (1/batch_size) * phi.T @ phi
        """
        X, y = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, d+1)

        # phi.T shape: (d+1, batch_size), phi shape: (batch_size, d+1) -> result (d+1, d+1)
        hessian = torch.matmul(phi.T, phi) / batch_size
        if return_grad:
            grad = self._grad_internal(phi, y, param, batch_size)
            return hessian, grad
        return hessian

    def hessian_column(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        param: torch.Tensor,
        columns: torch.Tensor,
        return_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute specific column(s) of the hessian, averaged over the batch. Assumes batched input.
        H_col = (1/batch_size) * phi.T @ phi[:, columns]
        """
        X, y = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, d+1)

        # phi.T shape: (d+1, batch_size), phi[:, col] shape: (batch_size,) -> result (d+1,)
        hessian_col = torch.matmul(phi.T, phi[:, columns]) / batch_size
        if return_grad:
            grad = self._grad_internal(phi, y, param, batch_size)
            return hessian_col, grad
        return hessian_col

    def hessian_vector(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        param: torch.Tensor,
        vector: torch.Tensor,
        return_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the Hessian-vector product for linear regression.
        """
        X, y = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, param_dim)

        if vector.ndim == 1:
            # Unsqueeze 1D vector to 2D (param_dim, 1) for consistent matrix multiplication
            vector_2d = vector.unsqueeze(1)
        elif vector.ndim == 2:
            vector_2d = vector

        phi_v = torch.matmul(phi, vector_2d)  # (batch_size, k)
        hessian_vector_product = torch.matmul(phi.T, phi_v) / batch_size  # (param_dim, k)
        if return_grad:
            grad = self._grad_internal(phi, y, param, batch_size)
            return hessian_vector_product, grad
        return hessian_vector_product

    def sherman_morrison(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        param: torch.Tensor,
        n_iter: int = None,
        return_grad: bool = False,
    ) -> torch.Tensor:
        """
        Compute the Sherman-Morrison term (feature vector phi), works only for a batch size of 1. Assumes batched input.
        """
        X, y = data
        # Ensure batch size is 1, assuming X already has batch dimension
        if X.size(0) != 1:
            raise ValueError("The Sherman-Morrison update is only possible for a batch size of 1")

        phi = self._add_bias(X)
        phi = phi.squeeze(0)  # Shape (d+1,)

        if return_grad:
            Y_pred = torch.matmul(phi, param)  # Shape (1,)
            error = Y_pred - y  # Shape (1,)
            grad = phi * error
            return phi, grad

        return phi  # The Sherman-Morrison term is just the feature vector phi


class LogisticRegression(BaseObjectiveFunction):
    """
    Logistic Regression class using PyTorch for Y in {0, 1}.
    Use Ridge regularization with lambda_ as the regularization parameter.
    """

    def __init__(self, bias: bool = True, lambda_: float = 0.0):
        self.bias = bias
        self.name = "Logistic Regression"
        self.lambda_ = lambda_

    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """
        Compute the logistic regression loss for Y in {0, 1}, assumes batched input.
        This is F(theta) = E[ -Y * log(sigma(X^T*theta)) - (1-Y) * log(1-sigma(X^T*theta)) ]
                    + (lambda/2) * ||theta||^2
        Returns the average loss over the batch.
        """
        X, y = data
        phi = self._add_bias(X)  # Shape (batch_size, d+1)
        dot_product = torch.matmul(phi, param)  # Shape (batch_size,)

        # Ensure y has the same shape as dot_product
        y_shaped = y.squeeze().view_as(dot_product)
        # Todo: Use BCEWithLogitsLoss for numerical stability.
        loss = torch.mean(torch.log(1 + torch.exp(dot_product)) - dot_product * y_shaped)

        if self.lambda_ > 0:
            loss += 0.5 * self.lambda_ * torch.norm(param, p=2) ** 2
        return loss

    def get_param_dim(self, data: Tuple[torch.Tensor, torch.Tensor]) -> int:
        """
        Return the dimension of the parameter vector.
        """
        X, _ = data
        feature_dim = X.shape[-1]
        return feature_dim + 1 if self.bias else feature_dim

        # grad = torch.matmul(phi.T, (sigmoid_dot_product - y)) / batch_size

    def _grad_internal(
        self, y: torch.Tensor, phi: torch.Tensor, sigmoid_dot_product: torch.Tensor, param: torch.Tensor
    ) -> torch.Tensor:
        """
        Helper to compute gradient from pre-computed values.
        grad = (phi.T @ (sigmoid(dot_product) - y)) / batch_size + lambda * param
        """
        batch_size = phi.size(0)
        # Ensure y has the same shape as sigmoid_dot_product
        y_shaped = y.squeeze().view_as(sigmoid_dot_product)
        grad = torch.matmul(phi.T, sigmoid_dot_product - y_shaped) / batch_size
        if self.lambda_ > 0:
            grad += self.lambda_ * param
        return grad

    def grad(self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the logistic regression loss, averaged over the batch. Assumes batched input.
        """
        X, y = data
        phi = self._add_bias(X)  # Shape (batch_size, d+1)
        dot_product = torch.matmul(phi, param)
        sigmoid_dot_product = torch.sigmoid(dot_product)

        grad = self._grad_internal(y, phi, sigmoid_dot_product, param)
        return grad

    def hessian(
        self, data: Tuple[torch.Tensor, torch.Tensor], param: torch.Tensor, return_grad: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the Hessian of the logistic regression loss, averaged over the batch. Assumes batched input.
        """
        X, y = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, d+1)
        dot_product = torch.matmul(phi, param)
        sigmoid_dot_product = torch.sigmoid(dot_product)

        weights = sigmoid_dot_product * (1 - sigmoid_dot_product)
        hessian = torch.matmul(phi.T * weights.unsqueeze(0), phi) / batch_size
        if self.lambda_ > 0:
            hessian += self.lambda_ * torch.eye(phi.shape[1], device=phi.device, dtype=phi.dtype)

        if return_grad:
            grad = self._grad_internal(y, phi, sigmoid_dot_product, param)
            return hessian, grad
        return hessian

    def hessian_column(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        param: torch.Tensor,
        columns: torch.Tensor,
        return_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute specific column(s) of the Hessian efficiently. Assumes columns is a 1D LongTensor."""
        X, y = data
        batch_size = X.size(0)
        phi = self._add_bias(X)  # Shape (batch_size, param_dim)
        dot_product = torch.matmul(phi, param)
        sigmoid_dot_product = torch.sigmoid(dot_product)  # Compute once

        weights = sigmoid_dot_product * (1 - sigmoid_dot_product)  # p(1-p), shape (batch_size,)

        if not columns.numel():  # if columns is empty, return an empty tensor
            hessian_cols = torch.empty((phi.shape[1], 0), device=phi.device, dtype=phi.dtype)
        else:
            phi_selected_cols = phi[:, columns]
            weighted_phi_selected_cols = weights.unsqueeze(1) * phi_selected_cols
            hessian_cols = torch.matmul(phi.T, weighted_phi_selected_cols) / batch_size
            if self.lambda_ > 0:
                num_selected_cols = columns.shape[0]
                if num_selected_cols > 0:
                    col_indices = torch.arange(num_selected_cols, device=phi.device)
                    hessian_cols[columns, col_indices] += self.lambda_

        if return_grad:
            grad = self._grad_internal(y, phi, sigmoid_dot_product, param)
            return hessian_cols, grad
        return hessian_cols

    def hessian_vector(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        param: torch.Tensor,
        vector: torch.Tensor,
        return_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the Hessian-vector product."""
        X, y = data
        batch_size = X.size(0)
        phi = self._add_bias(X)
        dot_product = torch.matmul(phi, param)
        sigmoid_dot_product = torch.sigmoid(dot_product)  # Compute once

        weights = sigmoid_dot_product * (1 - sigmoid_dot_product)  # p(1-p)

        weighted_phi_v = torch.matmul(phi, vector)  # (batch_size, k)
        weighted_phi_v = weights.unsqueeze(1) * weighted_phi_v  # (batch_size, k)
        hessian_v = torch.matmul(phi.T, weighted_phi_v) / batch_size  # (param_dim, k)

        if self.lambda_ > 0:
            hessian_v += self.lambda_ * vector  # (param_dim, k)

        if return_grad:
            grad = self._grad_internal(y, phi, sigmoid_dot_product, param)
            return hessian_v, grad
        return hessian_v
