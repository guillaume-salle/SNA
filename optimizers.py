import torch
from abc import ABC, abstractmethod
from typing import Tuple
import math
from objective_functions import BaseObjectiveFunction


class BaseOptimizer(ABC):
    """Base class for optimizers using PyTorch.

    This class provides a template for creating optimization algorithms.
    Subclasses should implement the `step` methods to define
    the specific behavior of the optimizer.

    Methods:
                step(data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> None:
                        Perform one optimization step. Should be implemented by subclasses.
    """

    DEFAULT_LOG_WEIGHT = 2.0
    DEFAULT_AVERAGED = False

    def __init__(
        self,
        param: torch.Tensor,
        obj_function: BaseObjectiveFunction,
        lr_exp: float,  # Required
        lr_const: float,  # Required
        lr_add: float,  # Required
        averaged: bool,
        batch_size: int,
        log_weight: float,
        device: torch.device,
    ) -> None:
        """
        Initialize the optimizer with parameters.
        Also initializes a non-averaged parameter, copy of the initial parameter if averaged.

        Args:
                param (torch.Tensor): The initial parameters for the optimizer.
                obj_function (BaseObjectiveFunction): The objective function to optimize.
                lr_exp (float): The exponent for the learning rate.
                lr_const (float): The constant for the learning rate.
                lr_add (float): The number of iterations to add to the learning rate.
                averaged (bool): Whether to use an averaged parameter.
                batch_size (int): The batch size size for optimization.
                log_weight (float): Exponent for the logarithmic weight.
                device (torch.device): The device to create tensors on.
        """
        self.device = device
        param = param.to(self.device)
        self.param = param
        self.obj_function = obj_function
        self.lr_exp = lr_exp
        self.lr_const = lr_const
        self.lr_add = lr_add
        self.batch_size = batch_size

        self.averaged = averaged
        self.log_weight = log_weight

        # Copy the initial parameter if averaged, otherwise use the same tensor reference
        # Use .clone().detach() for a non-gradient tracking copy
        self.param_not_averaged = param.clone().detach().to(self.device) if averaged else param
        if averaged:
            self.sum_weights = 0.0
        self.n_iter = 0

    @abstractmethod
    def step(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Perform one optimization step. Should be implemented by subclasses.
        """
        pass

    def update_averaged_param(self) -> None:
        """
        Update the averaged parameter using the current non-averaged parameter and the sum of weights.
        Uses Polyak-Ruppert averaging with optional logarithmic weighting.
        param = param + (weight / sum_weights) * (param_not_averaged - param)
        """
        if self.log_weight > 0:
            weight = math.log(self.n_iter + 1) ** self.log_weight
        else:
            weight = 1.0

        self.sum_weights += weight
        self.param += (weight / self.sum_weights) * (self.param_not_averaged - self.param)


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer using PyTorch.
    """

    name = "SGD"

    def __init__(
        self,
        param: torch.Tensor,
        obj_function: BaseObjectiveFunction,
        lr_exp: float,
        lr_const: float,
        lr_add: float,
        averaged: bool,
        batch_size: int,
        log_weight: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            param=param,
            obj_function=obj_function,
            lr_exp=lr_exp,
            lr_const=lr_const,
            lr_add=lr_add,
            averaged=averaged,
            batch_size=batch_size,
            log_weight=log_weight,
            device=device,
        )

    def step(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Perform one optimization step using PyTorch tensors.

        Parameters:
        data (torch.Tensor | Tuple[torch.Tensor, torch.Tensor]): The input data for the optimization step.
        """
        self.n_iter += 1
        grad = self.obj_function.grad(data, self.param_not_averaged)

        # Update the non averaged parameter
        learning_rate = self.lr_const / (self.n_iter**self.lr_exp + self.lr_add)
        self.param_not_averaged -= learning_rate * grad

        if self.averaged:
            self.update_averaged_param()


class mSNA(BaseOptimizer):
    """
    Masked Stochastic Newton Algorithm optimizer using PyTorch.
    """

    name = "mSNA"
    CONST_CONDITION = 1.0

    def __init__(
        self,
        param: torch.Tensor,
        obj_function: BaseObjectiveFunction,
        # required USNA specific parameters
        lr_hess_exp: float,
        lr_hess_const: float,
        lr_hess_add: float,
        averaged_matrix: bool,
        compute_hessian_param_avg: bool,
        proj: bool,
        # Base Optimizer parameters
        lr_exp: float,
        lr_const: float,
        lr_add: float,
        averaged: bool,
        batch_size: int,
        log_weight: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        device: torch.device = torch.device("cpu"),
        # other USNA specific parameters
        log_weight_matrix: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        version: str = "mask",
        mask_size: int = 1,
    ):
        # Initialize BaseOptimizer first to set up self.device, self.param etc.
        super().__init__(
            param=param,
            obj_function=obj_function,
            lr_exp=lr_exp,
            lr_const=lr_const,
            lr_add=lr_add,
            averaged=averaged,
            batch_size=batch_size,
            device=device,
            log_weight=log_weight,
        )

        self.lr_hess_exp = lr_hess_exp
        self.lr_hess_const = lr_hess_const
        self.lr_hess_add = lr_hess_add
        self.mask_size = min(mask_size, self.param.shape[0])
        self.averaged_matrix = averaged_matrix
        self.log_weight_matrix = log_weight_matrix
        self.compute_hessian_param_avg = compute_hessian_param_avg
        self.proj = proj
        self.version = version

        self.dim = self.param.shape[0]
        self.matrix = torch.eye(self.dim, device=self.device, dtype=self.param.dtype)
        self.matrix_not_avg = self.matrix.clone() if averaged_matrix else self.matrix
        if averaged_matrix:
            self.sum_weights_matrix = 0.0

        if version in ["mask", "original_mask"]:
            self.update_hessian = self.update_hessian_mask
        elif version in [
            "spherical_vector",
            "rademacher_vector",
            "original_spherical_vector",
            "original_rademacher_vector",
            "naive_spherical_vector",
            "naive_rademacher_vector",
            "orthogonal_vector",
        ]:
            self.update_hessian = self.update_hessian_vector
        elif version == "full":
            self.update_hessian = self.update_hessian_full
        else:
            raise ValueError(f"Invalid version: {version}")

        self.log_metrics_end = {"skip_update": 0}

    def step(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Perform one optimization step using PyTorch tensors.

        Args:
                data (torch.Tensor | Tuple[torch.Tensor, torch.Tensor]): The input data for the optimization step.
        """
        self.n_iter += 1

        # Update the hessian estimate and get the gradient from intermediate computation
        grad = self.update_hessian(data)
        if self.averaged_matrix:
            self.update_averaged_matrix()

        # Update theta
        learning_rate = self.lr_const / (self.n_iter**self.lr_exp + self.lr_add)
        self.param_not_averaged -= learning_rate * torch.matmul(self.matrix, grad)

        if self.averaged:
            self.update_averaged_param()

    def update_hessian_full(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Update the hessian estimate with the full hessian.
        """
        if self.compute_hessian_param_avg:
            hessian = self.obj_function.hessian(data, self.param)
            grad = self.obj_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian = self.obj_function.grad_and_hessian(data, self.param_not_averaged)

        lr_hessian = self.lr_hess_const / (self.n_iter**self.lr_hess_exp + self.lr_hess_add)

        # Log the norm of the hessian to see its behavior
        norm = torch.linalg.norm(hessian, ord=2)
        self.log_metrics = {}
        self.log_metrics["norm_hessian"] = norm

        if lr_hessian * norm < self.CONST_CONDITION:
            matrix_hessian = torch.matmul(self.matrix_not_avg, hessian)
            self.matrix_not_avg += -lr_hessian * (matrix_hessian + matrix_hessian.T) + lr_hessian**2 * torch.matmul(
                hessian, matrix_hessian
            )
            self.matrix_not_avg.diagonal().add_(2 * lr_hessian)
        else:
            self.log_metrics_end["skip_update"] += 1

        return grad

    def update_hessian_mask(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Update the hessian estimate with a canonic random vector using PyTorch, also returns grad.
        The Hessian estimate H is replaced by H M with M a mask matrix with mask_size ones on the diagonal.
        matrix_new = (I_d - lr *H M)^T A (I_d - lr *H M) + 2 * lr * M
        """
        # Generate masks. The mask matrix should be multiplied by (dim/mask_size) so it averages to the identity matrix,
        # but it leads to too much skipped updates. Hence we also divide the learning rate by (dim/mask_size) in theory,
        # and in practice lr / (dim/mask_size) * (dim/mask_size) * mask_matrix = lr * mask_matrix.
        masks = torch.randint(low=0, high=self.dim, size=(self.mask_size,), device=self.device)

        # Compute grad in the NOT averaged param, and hessian column in the desired param
        if self.compute_hessian_param_avg:
            hessian_columns = self.obj_function.hessian_column(data, self.param, masks)
            grad = self.obj_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian_columns = self.obj_function.grad_and_hessian_column(data, self.param_not_averaged, masks)

        lr_hessian = self.lr_hess_const / (self.n_iter**self.lr_hess_exp + self.lr_hess_add)

        if lr_hessian * torch.linalg.norm(hessian_columns, ord="fro") < self.CONST_CONDITION:
            # Compute this product only once and then transpose it
            matrix_hessian_vector = torch.matmul(self.matrix_not_avg, hessian_columns)

            self.matrix_not_avg[:, masks] -= lr_hessian * matrix_hessian_vector
            self.matrix_not_avg[masks, :] -= lr_hessian * matrix_hessian_vector.T
            if self.version == "mask":
                self.matrix_not_avg[masks[:, None], masks] += (lr_hessian**2) * torch.matmul(
                    hessian_columns.T, matrix_hessian_vector
                )
            elif self.version == "original_mask":
                pass
            else:
                raise ValueError(f"Invalid version for update_hessian_mask: {self.version}")
            # faster version
            # self.matrix_not_avg[masks, :] = self.matrix_not_avg[:, masks]  # second addition of diagonal terms done next
            # self.matrix_not_avg[masks[:, None], masks] += (lr_hessian**2) * torch.matmul(
            #     hessian_columns.T, product
            # ) - lr_hessian * product[masks]

            # Add efficiently to the diagonal
            if self.proj:  # add 2 * lr * mask_matrix
                self.matrix_not_avg.diagonal()[masks] += 2 * lr_hessian
            else:  # Add 2 * lr * E[V V^T] = 2 * lr * (mask_size/dim) * I
                self.matrix_not_avg.diagonal().add_(2 * lr_hessian * self.mask_size / self.dim)
        else:
            self.log_metrics_end["skip_update"] += 1

        return grad

    def update_hessian_vector(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Update the hessian estimate with a product of the hessian and a vector using PyTorch.
        Update: A_new = A_old - lr * (A_old * hess * V * V^T + V * V^T * hess^T * A_old)
                        + lr^2 * V * V^T * hess.T * A_old * hess * V * V^T + 2 * lr * P
                where P = V V^T if proj=True, else P = E[V V^T].
                The columns of V must be unit vectors.
        Update condition is norm(lr * H V V^T) or norm(lr * H V) < CONST_CONDITION using Frobenius norm as proxy.
        """
        # Generate random vector(s) V of shape (dim, mask_size)
        if self.version in ["spherical_vector", "original_spherical_vector", "naive_spherical_vector"]:
            vector_V = torch.randn(self.dim, self.mask_size, device=self.device, dtype=self.param.dtype)
            col_norms = torch.linalg.norm(vector_V, ord=2, dim=0, keepdim=True)
            vector_V = vector_V / (col_norms + 1e-12)  # Each column is unit norm
        elif self.version in ["rademacher_vector", "original_rademacher_vector", "naive_rademacher_vector"]:
            vector_V = torch.randint(
                low=0, high=2, size=(self.dim, self.mask_size), device=self.device, dtype=self.param.dtype
            )
            vector_V = 2 * vector_V - 1
            vector_V = vector_V / math.sqrt(self.dim)  # Each column is unit norm, E[v_i v_i^T] = I_d/dim
        elif self.version == "orthogonal_vector":
            # Generate a random orthogonal matrix using QR decomposition
            A = torch.randn(self.dim, self.dim, device=self.device, dtype=self.param.dtype)
            Q, _ = torch.linalg.qr(A)  # Q is orthogonal
            # Take first mask_size columns of Q
            vector_V = Q[:, : self.mask_size]  # Each column is already unit norm and orthogonal to others
        else:
            raise ValueError(f"Invalid version for update_hessian_vector: {self.version}")

        if self.compute_hessian_param_avg:
            hessian_V = self.obj_function.hessian_vector(data, self.param, vector_V)  # H @ V
            grad = self.obj_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian_V = self.obj_function.grad_and_hessian_vector(data, self.param_not_averaged, vector_V)

        lr_hessian = self.lr_hess_const / (self.n_iter**self.lr_hess_exp + self.lr_hess_add)

        perform_update = False
        if self.version in [
            "spherical_vector",
            "rademacher_vector",
            "original_spherical_vector",
            "original_rademacher_vector",
            "orthogonal_vector",
        ]:
            # Condition: lr * ||HV||_F < C
            if lr_hessian * torch.linalg.norm(hessian_V, ord="fro") < self.CONST_CONDITION:
                perform_update = True
        elif self.version in ["naive_spherical_vector", "naive_rademacher_vector"]:  # Naive
            # Condition: lr * ||HV||_op < C
            if lr_hessian * torch.linalg.norm(hessian_V, ord=2) < self.CONST_CONDITION:
                perform_update = True

        if perform_update:
            if self.version in [
                "spherical_vector",
                "rademacher_vector",
                "original_spherical_vector",
                "original_rademacher_vector",
                "orthogonal_vector",
            ]:  # Non-naive updates
                # A_new = A_old - lr(A_old H V V^T + V V^T H^T A_old) + lr^2 V (V^T H^T A_old H V) V^T + 2lr P
                A_H_v_vT = torch.linalg.multi_dot((self.matrix_not_avg, hessian_V, vector_V.T))  # A H V V^T

                self.matrix_not_avg -= lr_hessian * (A_H_v_vT + A_H_v_vT.T)

                if self.version in ["spherical_vector", "rademacher_vector", "orthogonal_vector"]:
                    # Term: V (V^T H^T A H V) V^T
                    term = torch.linalg.multi_dot((vector_V, hessian_V.T, A_H_v_vT))
                    self.matrix_not_avg += lr_hessian**2 * term
                elif self.version in ["original_spherical_vector", "original_rademacher_vector"]:
                    pass
                else:
                    raise ValueError(f"Invalid version for update_hessian_vector: {self.version}")

            elif self.version in ["naive_spherical_vector", "naive_rademacher_vector"]:  # Naive updates
                # A_new = A_old - lr(A_old M + M^T A_old) + lr^2 M^T A_old M + 2lr P, where M = HVV^T
                M = torch.matmul(hessian_V, vector_V.T)  # M = HVV^T
                A_M = torch.matmul(self.matrix_not_avg, M)  # A M

                self.matrix_not_avg -= lr_hessian * (A_M + A_M.T)
                self.matrix_not_avg += lr_hessian**2 * torch.matmul(M.T, A_M)  # M^T A M

            # Common projection part
            if self.proj:  # Add 2 * lr * V V^T
                self.matrix_not_avg += 2 * lr_hessian * torch.matmul(vector_V, vector_V.T)
            else:  # Add 2 * lr * E[V V^T] = 2 * lr * (mask_size/dim) * I
                self.matrix_not_avg.diagonal().add_(2 * lr_hessian * self.mask_size / self.dim)
        else:
            self.log_metrics_end["skip_update"] += 1

        return grad

    def update_averaged_matrix(self) -> None:
        """
        Update the averaged matrix using the current matrix and the sum of weights.
        """
        if self.log_weight_matrix > 0:
            weight_matrix = math.log(self.n_iter + 1) ** self.log_weight_matrix
        else:
            weight_matrix = 1.0
        self.sum_weights_matrix += weight_matrix
        self.matrix += (weight_matrix / self.sum_weights_matrix) * (self.matrix_not_avg - self.matrix)
