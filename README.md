# Stochastic Numerical Analysis (SNA) Experiment Framework

This framework is designed to run and benchmark various stochastic optimization algorithms on synthetic datasets, with a primary focus on linear regression problems. It features integration with Weights & Biases (WandB) for experiment tracking and visualization.

## Prerequisites

*   Python 3.9+
*   A Weights & Biases account (if you want to log results online).

## Setup

1.  **Clone the Repository (if applicable)**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Create a `requirements.txt` file with the following content:
    ```
    torch
    wandb
    PyYAML
    tqdm
    numpy # often a dependency of torch, good to have explicit
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

A recommended directory structure for your project:

```
.
├── .completed_runs.log       # Tracks completed runs to avoid re-computation
├── configs/
│   ├── problem/              # Example directory for problem configurations
│   │   └── linear_regression_problem.yaml
│   └── optimizers/           # Default directory for optimizer configurations
│       ├── SGD.yaml
│       ├── USNA.yaml
│       └── USNA_spherical.yaml
├── datasets.py               # Dataset generation logic
├── objective_functions.py    # Objective function definitions
├── optimizers.py             # Optimizer implementations
├── README.md                 # This file
├── run.py                    # Main script to run experiments
└── requirements.txt          # Python dependencies
```

## Configuration Files

Experiments are defined by two main types of YAML configuration files: problem configurations and optimizer configurations.

### 1. Problem Configuration

This file defines the dataset, the model, and other problem-specific parameters. It is passed to the script using the `-c` or `--config` argument.

**Example (`configs/problem/linear_regression_problem.yaml`):**
```yaml
# Problem-specific parameters
radius: 0.1              # Radius for initializing theta_init on a sphere around true_theta

# Model parameters (passed to the objective function class)
model_params:
  bias: True             # Whether the LinearRegression model includes a bias term

# Dataset parameters (passed to generate_linear_regression function)
dataset_params:
  n_dataset: 100000      # Total number of samples in the dataset
  param_dim: 10          # Dimension of the true_theta parameter vector (if true_theta not given)
  # true_theta: [0.5, 1.0, -0.2, ...] # Optionally provide the exact true_theta
  bias: True             # Whether the data generation process includes a bias feature
  cov_type: "random"     # Covariance matrix type for features X. Options:
                         # "identity", "toeplitz", "hard", "random"
  cov_const: 1.0         # Constant for covariance matrix (meaning depends on cov_type)
                         # For "toeplitz": decay constant (e.g., 1/d)
                         # For "hard": condition number like constant
                         # For "random": exponent for eigenvalue decay (1/j^cov_const)
  diag: False            # For "toeplitz": whether to modify the diagonal
  # data_batch_size is automatically set from optimizer_params.batch_size during run_experiment

# Name for the dataset, used in project naming on WandB
dataset: "LinearRegressionSynth"
```

**Key `problem_config` fields:**
*   `radius`: (float) Used to initialize the optimizer's parameter `theta_init` such that `||theta_init - true_theta||^2 = radius^2`.
*   `model_params`: (dict) Passed to the constructor of the model class (e.g., `LinearRegression`).
    *   `bias`: (bool) Whether the model itself should handle a bias term.
*   `dataset_params`: (dict) Parameters for data generation.
    *   `n_dataset`: (int) Total number of samples for the experiment.
    *   `param_dim`: (int) Dimension of the `true_theta` vector (if `true_theta` is not explicitly provided).
    *   `true_theta`: (list, optional) Explicitly define the true parameters. If given, `param_dim` is inferred.
    *   `bias`: (bool) Whether the generated features `phi` should include an intercept term.
    *   `cov_type`: (str) Type of covariance matrix for the features `X` (before bias is added).
        *   `identity`: Identity matrix.
        *   `toeplitz`: Toeplitz matrix with `rho^|i-j|`. `cov_const` here is `rho`.
        *   `hard`: A block-diagonal matrix designed to be ill-conditioned. `cov_const` influences conditioning.
        *   `random`: Matrix with random eigenvectors and eigenvalues decaying as `1/j^cov_const`.
    *   `cov_const`: (float, optional) Constant parameter for `cov_type`.
    *   `diag`: (bool, optional) For `cov_type: "toeplitz"`, whether to modify the diagonal.
*   `dataset`: (str) A descriptive name for the dataset, used in WandB project naming.

### 2. Optimizer Configurations

These files define the optimizer to be used and its parameters.
*   **Location**: By default, these files should be placed in the `configs/optimizers/` directory. This base path is defined by `OPTIMIZER_CONFIGS_DIR` in `run.py`.
*   **Naming**: When running `run.py`, you refer to these files by their name without the `.yaml` or `.yml` suffix (e.g., `-o SGD` will load `configs/optimizers/SGD.yaml`). If the file has a `.yml` extension, provide the name with `.yml` or ensure your default append logic in `run.py` also checks for `.yml`.

**Example (`configs/optimizers/SGD.yaml`):**
```yaml
optimizer: SGD
optimizer_params:
  lr_exp: 0.5
  lr_const: 0.1
  lr_add_iter: 100.0
  averaged: True
  log_weight: 2.0
  batch_size: 1        # Explicit batch size for SGD steps
  # batch_size_power: 1.0 # Alternative: batch_size = param_dim^batch_size_power
  multiply_lr: 0.0     # lr multiplier based on batch_size (0.0 = no multiplication)
  # device: "cuda"     # "cuda" or "cpu". Defaults to cuda if available, else cpu.
```

**Example (`configs/optimizers/USNA.yaml` - Base USNA config):**
```yaml
optimizer: USNA
optimizer_params:
  lr_exp: 0.67
  lr_const: 1.0
  lr_add_iter: 1.0
  averaged: True
  log_weight: 0.0

  lr_hess_exp: 0.67
  lr_hess_const: 1.0
  lr_hess_add_iter: 1.0
  averaged_matrix: True
  log_weight_matrix: 0.0
  compute_hessian_param_avg: False # Use param (averaged) or param_not_averaged for Hessian computations
  proj: True                     # Projection type for Hessian update
  version: "mask"                # Default USNA version ("mask", "spherical_vector", "rademacher_vector", "full")
  mask_size: 1

  batch_size_power: 1.0          # Default: batch_size = param_dim ^ batch_size_power
  # batch_size: 10 # Or specify explicitly
  multiply_lr: "default"         # "default" = 1.0 - lr_exp, or a float value
```

**Example (`configs/optimizers/USNA_spherical.yaml` - Inheriting and Overriding):**
```yaml
inherits_from: USNA.yaml  # Or just "USNA" if .yaml is appended by default

optimizer_params:
  version: spherical_vector
  # Other parameters from USNA.yaml are inherited unless overridden here
  # For example, to change learning rates for this specific version:
  # lr_exp: 0.5
```

**Key `optimizer_config` fields:**
*   `optimizer`: (str) Name of the optimizer class (e.g., "SGD", "USNA"). Must match a class returned by `get_optimizer_class` in `run.py`.
*   `optimizer_params`: (dict) Parameters passed to the optimizer's constructor.
    *   **Common Parameters (for `BaseOptimizer`)**:
        *   `lr_exp`: (float) Learning rate exponent `1/k^lr_exp`.
        *   `lr_const`: (float) Learning rate constant.
        *   `lr_add_iter`: (float) Additive term in learning rate denominator.
        *   `averaged`: (bool) Whether to use Polyak-Ruppert averaging for parameters.
        *   `log_weight`: (float, optional) Exponent for logarithmic weights if `averaged` is true. Default from `BaseOptimizer.DEFAULT_LOG_WEIGHT`.
        *   `batch_size`: (int, optional) Batch size for optimizer steps.
        *   `batch_size_power`: (float, optional) If `batch_size` is not given, it's calculated as `param_dim ** batch_size_power`. Default from `BaseOptimizer.DEFAULT_BATCH_SIZE_POWER`.
        *   `multiply_lr`: (float or str, optional) Multiplies `lr_const` by `batch_size ^ multiply_lr`. If "default", uses `1.0 - lr_exp`. Default is `0.0` (no multiplication).
        *   `device`: (str, optional) "cuda" or "cpu". If not specified, defaults to "cuda" if available, otherwise "cpu".
    *   **USNA-Specific Parameters**:
        *   `lr_hess_exp`, `lr_hess_const`, `lr_hess_add_iter`: Learning rate parameters for Hessian matrix updates.
        *   `averaged_matrix`: (bool) Whether to average the Hessian matrix estimate.
        *   `log_weight_matrix`: (float, optional) Logarithmic weight exponent for matrix averaging.
        *   `compute_hessian_param_avg`: (bool) If true, Hessian computations (e.g., `H(param,...)`) use the averaged parameter `param`; otherwise, they use `param_not_averaged`.
        *   `proj`: (bool) Projection type used in some Hessian update rules.
        *   `version`: (str) Version of USNA update: "mask", "spherical_vector", "rademacher_vector", "full".
        *   `mask_size`: (int) For `version: "mask"`, the number of Hessian columns/rows to sample.
*   `inherits_from`: (str, optional) Filename of a parent optimizer configuration to inherit from. The current file's parameters will deeply merge with and override the parent's.

## Running Experiments

The main script `run.py` is used to launch experiments.

**Command:**
```bash
python run.py -c <path_to_problem_config.yaml> -o <optimizer_name_1> [<optimizer_name_2> ...] -N <num_runs>
```

**Arguments:**
*   `-c, --config PATH`: **Required**. Path to the problem definition YAML file.
*   `-o, --optimizer NAME_OR_PATH [NAME_OR_PATH ...]`: **Required**.
    *   Name(s) of optimizer YAML file(s) (e.g., `SGD`, `USNA_spherical`) located in the directory specified by `OPTIMIZER_CONFIGS_DIR` in `run.py` (default: `configs/optimizers/`). The `.yaml` suffix is optional.
    *   Alternatively, you can use shell globbing to pass full paths, e.g., `configs/optimizers/USNA*.yaml`.
*   `-N, --N_runs INT`: Number of runs (seeds) for averaging. Default is `1`. Max is `100`.

**Examples:**
1.  Run SGD on `my_problem.yaml` for 10 seeds:
    ```bash
    python run.py -c configs/problem/my_problem.yaml -o SGD -N 10
    ```
2.  Run SGD and a specific USNA variant for 5 seeds each:
    ```bash
    python run.py -c configs/problem/my_problem.yaml -o SGD USNA_spherical -N 5
    ```
3.  Run all USNA variants whose config files start with `USNA` using shell globbing:
    ```bash
    python run.py -c configs/problem/my_problem.yaml -o configs/optimizers/USNA* -N 3
    ```

## Weights & Biases (WandB) Integration

Experiments are automatically logged to Weights & Biases if WandB is configured locally.

*   **Entity**: **IMPORTANT!** The script currently has the WandB entity hardcoded as `"USNA"` in the `wandb.init()` call within `run.py`. You **must** change this to your own WandB username or team name, or remove the `entity` argument if you want it to default to your logged-in WandB entity.
    ```python
    # In run.py, inside the seed loop:
    wandb_run = wandb.init(
        entity="YOUR_WANDB_ENTITY_HERE", # <-- CHANGE THIS
        project=project_name,
        # ...
    )
    ```

*   **Project Name**: Generated as `"{dataset_name}-{problem_hash}"`.
    *   `dataset_name` comes from the `dataset` field in the problem config.
    *   `problem_hash` is an MD5 hash of the sorted problem configuration string.
*   **Group Name**: Generated as `"{optimizer_config_filename_base}-{optimizer_hash}"`.
    *   `optimizer_config_filename_base` is the name of the optimizer config file (e.g., "SGD", "USNA_spherical").
    *   `optimizer_hash` is an MD5 hash of the sorted optimizer configuration string.
*   **Run Name**: Generated as `"{optimizer_config_filename_base}-{run_identifier}_seed{X}"`.
    *   `run_identifier` is an MD5 hash of the merged (problem + current optimizer) configuration string (first 6 characters).
    *   `seedX` is the current seed number.

**Viewing Results:**
On the WandB project page:
1.  Use the "Group" button/panel and group by `wandb_group` (which corresponds to the `group_name` defined above).
2.  Select the group(s) corresponding to your experiment series.
3.  In the chart settings (pencil icon), you can enable aggregation methods like 'Avg', 'StdDev', or 'Min/Max' across seeds for a given group.

## Output & Resuming

*   **Console Output**: The script prints progress, configuration details, and summaries to the console.
*   **Completion Log (`.completed_runs.log`)**:
    *   Each successfully completed run (a specific optimizer config + seed combination) has its unique run name logged to this file.
    *   On subsequent script executions, if a run name is found in this log, that specific run will be skipped, preventing re-computation.
    *   Delete this file if you want to re-run all experiments from scratch.

## Extending the Framework

### Adding New Optimizers

1.  Implement your new optimizer class in `optimizers.py`, ensuring it inherits from `BaseOptimizer`.
2.  Add your optimizer class to the `get_optimizer_class` function in `run.py`:
    ```python
    # In run.py
    def get_optimizer_class(optimizer_name: str) -> BaseOptimizer:
        if optimizer_name == "SGD":
            return SGD
        elif optimizer_name == "USNA":
            return USNA
        elif optimizer_name == "YourNewOptimizer": # Add this
            return YourNewOptimizerClass
        else:
            raise ValueError(f"Unknown optimizer specified in config: {optimizer_name}")
    ```
3.  Create a YAML configuration file for your new optimizer in the `configs/optimizers/` directory.

### Adding New Objective Functions or Datasets

1.  **Objective Functions**:
    *   Implement your new objective function class in `objective_functions.py`, inheriting from `BaseObjectiveFunction`.
    *   Update the `run_experiment` function in `run.py` if necessary to instantiate your new model type based on the problem configuration.
2.  **Datasets**:
    *   Implement your new dataset generation logic in `datasets.py`. This might involve creating a new `Dataset` or `IterableDataset` subclass.
    *   Update the `run_experiment` function in `run.py` to call your new data generation function based on parameters in the problem configuration.

---

This README should provide a good starting point for users of your framework. Remember to replace placeholders like `<your-repo-url>` and especially advise users to change the hardcoded WandB entity. 