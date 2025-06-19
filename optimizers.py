import torch
from abc import ABC, abstractmethod
from typing import Tuple
import math
from objective_functions import BaseObjectiveFunction
import torch.nn.functional as F
import traceback


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

        # IMPORTANT: When `averaged` is False, `self.param` is a direct reference
        # to `self.param_not_avg`. To maintain this link, all updates to
        # `self.param_not_avg` MUST be performed using in-place operations
        # (e.g., using .add_(), .sub_(), or other methods ending in '_').
        # Reassigning `self.param_not_avg` (e.g., `self.param_not_avg = ...`)
        # will break this reference, and `self.param` will no longer be updated.
        if averaged:
            # Use .clone().detach() for a non-gradient tracking copy
            self.param_not_avg = param.clone().detach()
            self.sum_weights = 0.0
        else:
            self.param_not_avg = param
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
        self.param.add_((self.param_not_avg - self.param), alpha=(weight / self.sum_weights))


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
        grad = self.obj_function.grad(data, self.param_not_avg)

        # Update the non averaged parameter
        learning_rate = self.lr_const / (self.n_iter**self.lr_exp + self.lr_add)
        self.param_not_avg.add_(grad, alpha=-learning_rate)

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
        # Params for initial hessian inversion
        init_hess_inv: bool = False,
        init_hess_inv_samples: int | None = None,
        init_hess_inv_reg: float = 1e-4,
    ):
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

        self.is_warming_up = init_hess_inv
        if self.is_warming_up:
            if init_hess_inv_samples is None:
                raise ValueError(
                    "If 'init_hess_inv' is True, 'init_hess_inv_samples' must be specified in the optimizer config."
                )
            if init_hess_inv_samples <= 0:
                raise ValueError(
                    f"'init_hess_inv_samples' must be a positive integer, but got {init_hess_inv_samples}."
                )
            print(f"   [mSNA] Optimizer entering warm-up phase to estimate initial Hessian.")
            print(f"   [mSNA] Will accumulate {init_hess_inv_samples} samples before computing Hessian.")
            self.warmup_samples_processed = 0
            self.init_hess_inv_samples = init_hess_inv_samples
            self.init_hess_inv_reg = init_hess_inv_reg
            self.warmup_data_batches = []

        # IMPORTANT: When averaged_matrix is False, self.matrix is a direct reference
        # to self.matrix_not_avg. To maintain this link, all updates to
        # self.matrix_not_avg MUST be performed using in-place operations
        # (e.g., using .copy_(), .add_(), or other methods ending in '_').
        # Reassigning self.matrix_not_avg (e.g., `self.matrix_not_avg = ...`)
        # will break this reference.
        self.matrix = torch.eye(self.dim, device=self.device, dtype=self.param.dtype)
        if averaged_matrix:
            self.matrix_avg = torch.eye(self.dim, device=self.device, dtype=self.param.dtype)
            self.sum_weights_matrix = 0.0
        else:
            self.matrix_avg = self.matrix

        if version in ["mask", "mask_USNA"]:
            self.update_hessian = self.update_hessian_mask
        elif version in [
            "spherical_vector",
            "rademacher_vector",
            "spherical_vector_USNA",
            "rademacher_vector_USNA",
            "orthogonal_vector",
            "orthogonal_vector_USNA",
        ]:
            self.update_hessian = self.update_hessian_vector
        elif version in ["full", "full_USNA"]:
            self.update_hessian = self.update_hessian_full
        else:
            raise ValueError(f"Invalid version: {version}")

        self.log_metrics_end = {"skip_update": 0}

    def _complete_warmup(self):
        """Finalizes the warm-up phase by computing and setting the initial matrix."""
        print(f"   [mSNA] Warm-up complete. Processed {self.warmup_samples_processed} samples.")
        if not self.warmup_data_batches:
            print("   [mSNA] WARNING: No samples accumulated during warm-up. Using identity matrix.")
        else:
            try:
                # --- Generic Data Concatenation ---
                first_batch = self.warmup_data_batches[0]
                warmup_data_for_hessian: torch.Tensor | Tuple[torch.Tensor, ...]

                if isinstance(first_batch, tuple):
                    # Data is a tuple of tensors, e.g., (X, y)
                    # We need to concatenate each part of the tuple separately
                    num_tensors_in_batch = len(first_batch)
                    concatenated_tensors = []
                    for i in range(num_tensors_in_batch):
                        tensors_to_cat = [batch[i] for batch in self.warmup_data_batches]
                        concatenated_tensors.append(torch.cat(tensors_to_cat, dim=0))
                    warmup_data_for_hessian = tuple(concatenated_tensors)
                    print(
                        f"   [mSNA] Computing Hessian on accumulated data tuple. First tensor shape: {concatenated_tensors[0].shape}. This may use significant memory."
                    )
                else:
                    # Data is a single tensor
                    warmup_data_for_hessian = torch.cat(self.warmup_data_batches, dim=0)
                    print(
                        f"   [mSNA] Computing Hessian on accumulated data tensor of shape {warmup_data_for_hessian.shape}. This may use significant memory."
                    )

                # Compute Hessian once on the full accumulated data.
                # The objective function's hessian method already averages by batch size.
                avg_hessian = self.obj_function.hessian(warmup_data_for_hessian, self.param)

                # Regularize before inverting
                avg_hessian.diagonal().add_(self.init_hess_inv_reg)

                initial_matrix = torch.linalg.inv(avg_hessian)

                self.matrix.copy_((initial_matrix + initial_matrix.T) / 2)  # Symmetrize
                if self.averaged_matrix:
                    self.matrix_avg.copy_(self.matrix)

                # Optional: Print the eigenvalues of the initial matrix
                print("   [mSNA] Successfully initialized matrix with inverse Hessian estimate.")
                eigenvalues = torch.linalg.eigvalsh(self.matrix)
                print(
                    f"   [mSNA] Initial matrix condition number: {eigenvalues.max()/eigenvalues.min():.2e} (min={eigenvalues.min():.4f}, max={eigenvalues.max():.4f})"
                )

            except Exception as e:
                print(
                    f"   [mSNA] WARNING: Failed to compute or invert Hessian during warm-up: {e}. Falling back to Identity matrix."
                )
                traceback.print_exc()

        # Clean up warm-up attributes to free memory
        del self.warmup_data_batches
        del self.warmup_samples_processed
        del self.init_hess_inv_samples
        del self.init_hess_inv_reg
        self.is_warming_up = False

        # If matrix averaging is on, reset the average matrix to the new initial matrix
        if self.averaged_matrix:
            self.matrix_avg.copy_(self.matrix)
            self.sum_weights_matrix = 0.0

    def step(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Perform one optimization step using PyTorch tensors.
        Includes a warm-up phase for initial Hessian estimation if configured.
        """
        self.log_metrics = {}
        # --- Warm-up Phase ---
        if self.is_warming_up:
            # Determine batch size from the first tensor in data
            current_batch_size = data[0].size(0) if isinstance(data, tuple) else data.size(0)

            # Accumulate data batches
            self.warmup_data_batches.append(data)
            self.warmup_samples_processed += current_batch_size

            # Check if warm-up is complete
            if self.warmup_samples_processed >= self.init_hess_inv_samples:
                self._complete_warmup()

            # During warm-up, we do not update parameters, so we return here.
            # self.n_iter is also not incremented yet.
            return

        # --- Normal Optimization Step ---
        self.n_iter += 1

        # Update the hessian estimate and get the gradient from intermediate computation
        grad = self.update_hessian(data)
        if self.averaged_matrix:
            self.update_averaged_matrix()

        # Optional : Assert that the matrix remains symmetric after the update.
        # This is crucial for numerical stability.
        if self.n_iter % 1000 == 0:
            assert torch.allclose(self.matrix, self.matrix.T), "Matrix lost symmetry, indicating numerical instability."

        # Update theta
        learning_rate = self.lr_const / (self.n_iter**self.lr_exp + self.lr_add)
        if self.averaged_matrix:
            direction = torch.matmul(self.matrix_avg, grad)
        else:
            direction = torch.matmul(self.matrix, grad)
        self.param_not_avg.add_(direction, alpha=-learning_rate)

        if self.averaged:
            self.update_averaged_param()

    def update_hessian_full(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Update the hessian estimate with the full hessian.
        Update: A_new = A - lr * ( (AH + HA)/2 - I) + (lr/2)^2 * (HAH)
        or
        A_new = (I - lr/2 * H) A (I - lr/2 * H) + lr * I
        """
        include_lr_squared_term = "USNA" not in self.version

        if self.compute_hessian_param_avg:
            hessian = self.obj_function.hessian(data, self.param)
            grad = self.obj_function.grad(data, self.param_not_avg)
        else:
            hessian, grad = self.obj_function.hessian(data, self.param_not_avg, return_grad=True)

        lr_hessian = self.lr_hess_const / (self.n_iter**self.lr_hess_exp + self.lr_hess_add)

        norm_hessian = torch.linalg.norm(hessian, ord=2)

        # Log the norm of the hessian to see its behavior
        self.log_metrics["norm_hessian"] = norm_hessian

        if (lr_hessian / 2) * norm_hessian < self.CONST_CONDITION:
            # --- TESTING CONFIGURATION ---
            # Set to True to use the expanded form, False to use the direct/compact form.
            USE_EXPANDED_FORM = True

            if USE_EXPANDED_FORM:
                # --- Version 1: Expanded Form ---
                # A_new = A - lr(AH + HA) + lr^2(HAH)
                # This form is analogous to the update in `update_hessian_mask`.
                matrix_hessian = torch.matmul(self.matrix, hessian)
                self.matrix.add_(matrix_hessian + matrix_hessian.T, alpha=-lr_hessian / 2)
                if include_lr_squared_term:
                    temp_matrix = torch.matmul(hessian, matrix_hessian)
                    # Explicitly symmetrize to prevent instability.
                    self.matrix.add_(temp_matrix + temp_matrix.T, alpha=(lr_hessian / 2) ** 2 / 2)
            else:
                # --- Version 2: Product Form ---
                if include_lr_squared_term:
                    term_factor = (
                        torch.eye(self.dim, device=self.device, dtype=self.param.dtype) - (lr_hessian / 2) * hessian
                    )
                    temp_matrix = term_factor @ self.matrix @ term_factor
                else:
                    term_factor = torch.eye(self.dim, device=self.device, dtype=self.param.dtype) - lr_hessian * hessian
                    temp_matrix = term_factor @ self.matrix
                # Explicitly symmetrize the result to counteract floating-point inaccuracies
                # that can break perfect symmetry in matrix multiplication.
                temp_matrix = (temp_matrix + temp_matrix.T) / 2
                self.matrix.copy_(temp_matrix)

            # Common added term for all versions
            self.matrix.diagonal().add_(lr_hessian)

        else:
            self.log_metrics_end["skip_update"] += 1

        # Optional, log the norm of A_n
        # norm_An = torch.linalg.norm(self.matrix, ord="fro")
        # norm_An_op = torch.linalg.norm(self.matrix, ord=2)
        # self.log_metrics["norm_An"] = norm_An
        # self.log_metrics["norm_An_op"] = norm_An_op

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
        include_lr_squared_term = "USNA" not in self.version
        lr_hessian = self.lr_hess_const / (self.n_iter**self.lr_hess_exp + self.lr_hess_add)

        if self.mask_size == 1:
            # --- Optimized path for mask_size = 1 ---
            mask = torch.randint(low=0, high=self.dim, size=(1,), device=self.device).item()

            if self.compute_hessian_param_avg:
                hessian_column = self.obj_function.hessian_column(
                    data, self.param, torch.tensor([mask], device=self.device)
                ).squeeze()
                grad = self.obj_function.grad(data, self.param_not_avg)
            else:
                hessian_column, grad = self.obj_function.hessian_column(
                    data, self.param_not_avg, torch.tensor([mask], device=self.device), return_grad=True
                )
                hessian_column = hessian_column.squeeze()

            norm_hessian = torch.linalg.vector_norm(hessian_column)
            self.log_metrics["norm_hessian"] = norm_hessian

            if (lr_hessian / 2) * norm_hessian < self.CONST_CONDITION:
                # Let c = A h_i. All operations are on vectors, which is faster.
                c_vector = torch.matmul(self.matrix, hessian_column)

                # First-order update: A_new = A - lr(c*e_i^T + e_i*c^T)
                # Update column i, then row i. The diagonal element is correctly updated twice.
                self.matrix[:, mask].add_(c_vector, alpha=-lr_hessian / 2)
                self.matrix[mask, :] = self.matrix[:, mask]  # no need to transpose a vector
                self.matrix[mask, mask] -= (lr_hessian / 2) * c_vector[mask]

                # Second-order update (if applicable)
                if include_lr_squared_term:
                    update_scalar = torch.dot(hessian_column, c_vector)
                    self.matrix[mask, mask].add_(update_scalar, alpha=(lr_hessian / 2) ** 2)

                # Added identity term
                if self.proj:
                    self.matrix[mask, mask].add_(lr_hessian)
                else:
                    self.matrix.diagonal().add_(lr_hessian / self.dim)
            else:
                self.log_metrics_end["skip_update"] += 1
            return grad

        else:
            # --- General path for mask_size > 1 ---
            # Generate unique random indices by taking the first `mask_size` elements of a random permutation.
            shuffled_indices = torch.randperm(self.dim, device=self.device)
            masks = shuffled_indices[: self.mask_size]

            if self.compute_hessian_param_avg:
                hessian_columns = self.obj_function.hessian_column(data, self.param, masks)
                grad = self.obj_function.grad(data, self.param_not_avg)
            else:
                hessian_columns, grad = self.obj_function.hessian_column(
                    data, self.param_not_avg, masks, return_grad=True
                )

            norm_hessian = torch.linalg.norm(hessian_columns, ord=2)
            self.log_metrics["norm_hessian"] = norm_hessian

            if (lr_hessian / 2) * norm_hessian < self.CONST_CONDITION:
                # --- 1. First-Order Update: A_new = A - lr(AHM + MHA) ---
                matrix_hessian_columns = torch.matmul(self.matrix, hessian_columns)
                # Update columns
                self.matrix.index_add_(1, masks, matrix_hessian_columns, alpha=-lr_hessian / 2)
                # Update non diagonal row block by copying
                all_indices = torch.arange(self.dim)
                not_masks_bool = torch.ones(self.dim, dtype=torch.bool)
                not_masks_bool[masks] = False
                not_masks = all_indices[not_masks_bool]
                self.matrix[masks[:, None], not_masks] = self.matrix[not_masks, masks[:, None]]
                # Add also the transpose for the diagonal block
                self.matrix[masks[:, None], masks] -= (lr_hessian / 2) * matrix_hessian_columns[masks, :].T

                # --- 2. Second-Order Update (if applicable) ---
                if include_lr_squared_term:
                    # The second-order term is lr^2 * M H A H M = lr^2 * (H M)^T A (H M).
                    # This only affects the A[masks, masks] block.
                    term = torch.matmul(hessian_columns.T, matrix_hessian_columns)
                    self.matrix[masks[:, None], masks] += (lr_hessian / 2) ** 2 * term

                # Explicitly symmetrize the diagonal bloc to ensure symmetry
                diag_bloc = self.matrix[masks[:, None], masks]
                self.matrix[masks[:, None], masks] = (diag_bloc + diag_bloc.T) / 2

                # --- 3. Identity Term ---
                if self.proj:  # add lr * mask_matrix
                    diag_update = torch.full(masks.shape, lr_hessian, device=self.device, dtype=self.param.dtype)
                    self.matrix.diagonal().index_add_(0, masks, diag_update)
                else:  # Add lr * E[V V^T] = lr * (mask_size/dim) * I
                    self.matrix.diagonal().add_(lr_hessian * self.mask_size / self.dim)
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
        include_lr_squared_term = "USNA" not in self.version
        vector_type = ""
        if "spherical" in self.version:
            vector_type = "spherical"
        elif "rademacher" in self.version:
            vector_type = "rademacher"
        elif "orthogonal" in self.version:
            vector_type = "orthogonal"

        # Generate random vector(s) V of shape (dim, mask_size)
        if vector_type == "spherical":
            vector_V = torch.randn(self.dim, self.mask_size, device=self.device, dtype=self.param.dtype)
            col_norms = torch.linalg.norm(vector_V, ord=2, dim=0, keepdim=True)
            vector_V = vector_V / (col_norms + 1e-12)  # Each column is unit norm
        elif vector_type == "rademacher":
            vector_V = torch.randint(
                low=0, high=2, size=(self.dim, self.mask_size), device=self.device, dtype=self.param.dtype
            )
            vector_V = 2 * vector_V - 1
            vector_V = vector_V / math.sqrt(self.dim)  # Each column is unit norm, E[v_i v_i^T] = I_d/dim
        elif vector_type == "orthogonal":
            # Generate a random orthogonal matrix using QR decomposition
            A = torch.randn(self.dim, self.dim, device=self.device, dtype=self.param.dtype)
            Q, _ = torch.linalg.qr(A)  # Q is orthogonal
            vector_V = Q[:, : self.mask_size]
        else:
            raise ValueError(f"Invalid version for update_hessian_vector: {self.version}")

        if self.compute_hessian_param_avg:
            hessian_V = self.obj_function.hessian_vector(data, self.param, vector_V)
            grad = self.obj_function.grad(data, self.param_not_avg)
        else:
            hessian_V, grad = self.obj_function.hessian_vector(data, self.param_not_avg, vector_V, return_grad=True)

        lr_hessian = self.lr_hess_const / (self.n_iter**self.lr_hess_exp + self.lr_hess_add)

        norm_hessian = torch.linalg.norm(hessian_V, ord=2)
        self.log_metrics["norm_hessian"] = norm_hessian

        if (lr_hessian / 2) * norm_hessian < self.CONST_CONDITION:
            # A_new = A_old - lr(A_old H V V^T + V V^T H^T A_old) + lr^2 V (V^T H^T A_old H V) V^T + 2lr P
            A_H_v_vT = torch.linalg.multi_dot((self.matrix, hessian_V, vector_V.T))  # A H V V^T
            self.matrix.add_((A_H_v_vT + A_H_v_vT.T), alpha=-lr_hessian / 2)

            if include_lr_squared_term:
                # Term: V (V^T H^T A H V) V^T
                temp_matrix = torch.linalg.multi_dot((vector_V, hessian_V.T, A_H_v_vT))
                self.matrix.add_(temp_matrix + temp_matrix.T, alpha=(lr_hessian / 2) ** 2)

            if self.proj:  # Add lr * V V^T
                self.matrix.add_(torch.matmul(vector_V, vector_V.T), alpha=lr_hessian)
            else:  # Add 2 * lr * E[V V^T] = 2 * lr * (mask_size/dim) * I
                self.matrix.diagonal().add_(2 * lr_hessian * self.mask_size / self.dim)
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
        self.matrix_avg.add_((self.matrix - self.matrix_avg), alpha=(weight_matrix / self.sum_weights_matrix))


class SNA(BaseOptimizer):
    """
    Stochastic Noewton Algorithm optimizer using PyTorch.
    """

    name = "SNA"

    def __init__(
        self,
        param: torch.Tensor,
        obj_function: BaseObjectiveFunction,
        # required SNA specific parameters
        init_id_weight: float,
        compute_hessian_param_avg: bool,
        # Base Optimizer parameters
        lr_exp: float,
        lr_const: float,
        lr_add: float,
        averaged: bool,
        batch_size: int,
        log_weight: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        device: torch.device = torch.device("cpu"),
        # other SNA specific parameters
        log_weight_matrix: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
    ):
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

        self.init_id_weight = init_id_weight
        self.compute_hessian_param_avg = compute_hessian_param_avg
        self.log_weight_matrix = log_weight_matrix

        self.dim = self.param.shape[0]
        self.hessian_bar = torch.eye(self.dim, device=self.device, dtype=self.param.dtype)
        self.sum_weights_hessian = self.init_id_weight
        self.matrix = torch.eye(self.dim, device=self.device, dtype=self.param.dtype)

    def step(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Perform one optimization step using PyTorch tensors.

        Args:
                data (torch.Tensor | Tuple[torch.Tensor, torch.Tensor]): The input data for the optimization step.
        """
        self.log_metrics = {}
        self.n_iter += 1

        # Update the hessian estimate and get the gradient from intermediate computation
        grad = self.update_hessian(data)

        # Update theta
        learning_rate = self.lr_const / (self.n_iter**self.lr_exp + self.lr_add)
        self.param_not_avg.add_(torch.matmul(self.matrix, grad), alpha=-learning_rate)

        if self.averaged:
            self.update_averaged_param()

    def update_hessian(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Update the hessian estimate using the current hessian and the sum of weights.
        """
        if self.compute_hessian_param_avg:
            # Compute hessian at the averaged parameter for stability
            hessian = self.obj_function.hessian(data, self.param)
            # Compute gradient at the non-averaged parameter for the update step
            grad = self.obj_function.grad(data, self.param_not_avg)
        else:
            # Compute both hessian and gradient at the non-averaged parameter
            hessian, grad = self.obj_function.hessian(data, self.param_not_avg, return_grad=True)

        if self.log_weight_matrix > 0:
            weight = math.log(self.n_iter + 1) ** self.log_weight_matrix
        else:
            weight = 1.0

        self.sum_weights_hessian += weight
        self.hessian_bar.add_((hessian - self.hessian_bar), alpha=(weight / self.sum_weights_hessian))
        self.matrix = torch.linalg.inv(self.hessian_bar)

        norm_hessian = torch.linalg.norm(hessian, ord=2)
        self.log_metrics["norm_hessian"] = norm_hessian
        norm_hessian_bar = torch.linalg.norm(self.hessian_bar, ord=2)
        self.log_metrics["SNA_norm_hessian_bar"] = norm_hessian_bar
        norm_matrix = torch.linalg.norm(self.matrix, ord=2)
        self.log_metrics["SNA_norm_matrix"] = norm_matrix

        return grad
