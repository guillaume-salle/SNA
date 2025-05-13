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
    DEFAULT_BATCH_SIZE_POWER = 1.0
    DEFAULT_AVERAGED = False
    DEFAULT_MULTIPLY_LR = 0.0

    def __init__(
        self,
        param: torch.Tensor,
        obj_function: BaseObjectiveFunction,
        lr_exp: float,  # Required
        lr_const: float,  # Required
        lr_add_iter: float,  # Required
        averaged: bool,
        log_weight: float | None,
        batch_size: int | None,
        batch_size_power: float,
        multiply_lr: float | str | None,  # if str use "default". 0 for no multiplication
        device: str,
    ) -> None:
        """
        Initialize the optimizer with parameters.
        Also initializes a non-averaged parameter, copy of the initial parameter if averaged.

        Args:
                param (torch.Tensor): The initial parameters for the optimizer.
                obj_function (BaseObjectiveFunction): The objective function to optimize.
                lr_exp (float): The exponent for the learning rate.
                lr_const (float): The constant for the learning rate.
                lr_add_iter (float): The number of iterations to add to the learning rate.
                averaged (bool): Whether to use an averaged parameter.
                log_weight (float): Exponent for the logarithmic weight.
                batch_size (int | None): The batch size size for optimization. If None, calculated from the power of the dimension.
                batch_size_power (float): The power of the dimension for the batch size.
                multiply_lr (float | str): Multiply the learning rate by batch_size^multiply_lr, for mini-batch. 'default' uses 1 - lr_exp.
                device (str): The device to create tensors on ('cpu' or 'cuda').
        """
        self.device = device
        param = param.to(self.device)
        self.param = param
        self.obj_function = obj_function
        self.lr_exp = lr_exp
        self.lr_const = lr_const
        self.lr_add_iter = lr_add_iter

        # Batch size is either given or if not, calculated from the power of the dimension
        self.batch_size = batch_size if batch_size is not None else int(param.shape[0] ** batch_size_power)

        self.averaged = averaged
        self.log_weight = log_weight

        # Multiply the learning rate by an exponent of the batch_size
        if isinstance(multiply_lr, str) and multiply_lr == "default":
            multiply_lr = 1.0 - self.lr_exp
        if multiply_lr > 0 and self.batch_size > 1:
            self.lr_const *= self.batch_size**multiply_lr

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
        lr_add_iter: float,
        averaged: bool = False,
        log_weight: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        batch_size: int | None = None,
        batch_size_power: float = BaseOptimizer.DEFAULT_BATCH_SIZE_POWER,
        multiply_lr: float | str = 0.0,
        device: str | None = None,
    ):
        super().__init__(
            param=param,
            obj_function=obj_function,
            batch_size=batch_size,
            batch_size_power=batch_size_power,
            lr_exp=lr_exp,
            lr_const=lr_const,
            lr_add_iter=lr_add_iter,
            averaged=averaged,
            log_weight=log_weight,
            multiply_lr=multiply_lr,
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
        learning_rate = self.lr_const / (self.n_iter**self.lr_exp + self.lr_add_iter)
        self.param_not_averaged -= learning_rate * grad

        if self.averaged:
            self.update_averaged_param()


class USNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer using PyTorch.
    """

    name = "USNA"
    CONST_CONDITION = 0.5

    def __init__(
        self,
        param: torch.Tensor,
        obj_function: BaseObjectiveFunction,
        # required USNA specific parameters
        lr_hess_exp: float,
        lr_hess_const: float,
        lr_hess_add_iter: float,
        averaged_matrix: bool,
        compute_hessian_param_avg: bool,
        proj: bool,
        # Base Optimizer parameters
        lr_exp: float,
        lr_const: float,
        lr_add_iter: float,
        averaged: bool = False,
        log_weight: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        batch_size: int | None = None,
        batch_size_power: float = BaseOptimizer.DEFAULT_BATCH_SIZE_POWER,
        multiply_lr: float | str = 0.0,
        device: str | None = None,
        # other USNA specific parameters
        log_weight_matrix: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        version: str = "mask",
        mask_size: int = 1,
    ):
        # Initialize BaseOptimizer first to set up self.device, self.param etc.
        super().__init__(
            param=param,
            obj_function=obj_function,
            batch_size=batch_size,
            batch_size_power=batch_size_power,
            lr_exp=lr_exp,
            lr_const=lr_const,
            lr_add_iter=lr_add_iter,
            averaged=averaged,
            log_weight=log_weight,
            multiply_lr=multiply_lr,
            device=device,
        )

        self.lr_hess_exp = lr_hess_exp
        self.lr_hess_const = lr_hess_const
        self.lr_hess_add_iter = lr_hess_add_iter
        self.mask_size = mask_size
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

        if version == "mask":
            self.update_hessian = self.update_hessian_mask
        elif version == "spherical_vector" or version == "rademacher_vector":
            self.update_hessian = self.update_hessian_vector
        elif version == "full":
            self.update_hessian = self.update_hessian_full
        else:
            raise ValueError(f"Invalid version: {version}")

        self.log_metrics = {"skip_update": 0}

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
        learning_rate = self.lr_const / (self.n_iter**self.lr_exp + self.lr_add_iter)
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

        lr_hessian = self.lr_hess_const / (self.n_iter**self.lr_hess_exp + self.lr_hess_add_iter)

        if torch.linalg.norm(hessian, ord="fro") <= self.CONST_CONDITION / lr_hessian:
            product = torch.matmul(self.matrix_not_avg, hessian)
            self.matrix_not_avg += -lr_hessian * (product + product.T) + lr_hessian**2 * torch.matmul(hessian, product)
            self.matrix_not_avg.diagonal().add_(2 * lr_hessian)
        else:
            self.log_metrics["skip_update"] += 1

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
        # Remember to divide lr by (dim/mask_size) when adding identity.
        masks = torch.randint(low=0, high=self.dim, size=(self.mask_size,))

        # Compute grad in the NOT averaged param, and hessian column in the desired param
        if self.compute_hessian_param_avg:
            hessian_columns = self.obj_function.hessian_column(data, self.param, masks)
            grad = self.obj_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian_columns = self.obj_function.grad_and_hessian_column(data, self.param_not_averaged, masks)

        lr_hessian = self.lr_hess_const / (self.n_iter**self.lr_hess_exp + self.lr_hess_add_iter)

        if torch.linalg.norm(hessian_columns, ord="fro") <= self.CONST_CONDITION / lr_hessian:
            # Compute this product only once and then transpose it
            product = torch.matmul(self.matrix_not_avg, hessian_columns)

            self.matrix_not_avg[:, masks] -= lr_hessian * product
            self.matrix_not_avg[masks, :] -= lr_hessian * product.T
            self.matrix_not_avg[masks[:, None], masks] += (lr_hessian**2) * torch.matmul(hessian_columns.T, product)
            # faster version
            # self.matrix_not_avg[masks, :] = self.matrix_not_avg[:, masks]  # second addition of diagonal terms done next
            # self.matrix_not_avg[masks[:, None], masks] += (lr_hessian**2) * torch.matmul(
            #     hessian_columns.T, product
            # ) - lr_hessian * product[masks]

            # Add efficiently to the diagonal
            if self.proj:  # add 2 * lr * mask_matrix
                self.matrix_not_avg.diagonal()[masks] += 2 * lr_hessian
            else:  # add 2 * lr *Id, but lr should be lr / (dim / mask_size)
                self.matrix_not_avg.diagonal().add_(2 * lr_hessian / (self.dim / self.mask_size))
        else:
            self.log_metrics["skip_update"] += 1

        return grad

    def update_hessian_vector(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Update the hessian estimate with a product of the hessian and a vector using PyTorch.
        Update: A_new = A_old - lr * (A_old * hess * v * v^T + v * v^T * hess^T * A_old)
                        + lr^2 * v * v^T * hess.T * A_old * hess * v * v^T + 2 * lr * v * v^T
                      = (I_d - lr * hess * v * v^T)^T @ A_old @ (I_d - lr * hess * v * v^T) + 2 * lr * v * v^T
        """
        if self.version == "spherical_vector":  # on the sphere of radius sqrt(d)
            vector = torch.randn(self.dim, device=self.device, dtype=self.param.dtype)
            vector = math.sqrt(self.dim) * vector / torch.linalg.norm(vector)
        elif self.version == "rademacher_vector":
            vector = torch.randint(low=0, high=2, size=(self.dim,), device=self.device, dtype=self.param.dtype)
            vector = 2 * vector - 1

        if self.compute_hessian_param_avg:
            hessian_vector = self.obj_function.hessian_vector(data, self.param, vector)
            grad = self.obj_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian_vector = self.obj_function.grad_and_hessian_vector(data, self.param_not_averaged, vector)

        lr_hessian = self.lr_hess_const / (self.n_iter**self.lr_hess_exp + self.lr_hess_add_iter)

        if torch.linalg.norm(hessian_vector) <= self.CONST_CONDITION / lr_hessian:
            matrix_hessian_vector = torch.matmul(self.matrix_not_avg, hessian_vector)
            outer_product = torch.outer(matrix_hessian_vector, vector)
            self.matrix_not_avg -= lr_hessian * (outer_product + outer_product.T)
            self.matrix_not_avg += lr_hessian**2 * torch.outer(vector, torch.matmul(hessian_vector.T, outer_product))

            if self.proj:  # add 2 * lr * vector @ vector.T
                self.matrix_not_avg.add_(2 * lr_hessian * torch.outer(vector, vector))
            else:  # add 2 * lr * I_d
                self.matrix_not_avg.diagonal().add_(2 * lr_hessian)
        else:
            self.log_metrics["skip_update"] += 1

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
