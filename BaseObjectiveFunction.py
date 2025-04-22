from abc import ABC, abstractmethod
import torch


class BaseObjectiveFunction(ABC):
    def __init__(self, X, y, theta_true=None):
        self.X = X
        self.y = y
        self.theta_true = theta_true
        self.n_samples, self.n_features = X.shape
        self.device = X.device  # Assume tensors are on same device

    @abstractmethod
    def evaluate(self, theta):
        """Calculates the loss."""
        pass

    @abstractmethod
    def gradient(self, theta):
        """Calculates the gradient."""
        pass

    @abstractmethod
    def hessian_vector_product(self, theta, v):
        """Calculates the Hessian-vector product H*v."""
        pass

    def calculate_error_norm(self, theta):
        """Calculates L2 norm ||theta - theta_true|| if theta_true is known."""
        if self.theta_true is not None:
            with torch.no_grad():
                return torch.linalg.norm(theta - self.theta_true).item()
        return None
