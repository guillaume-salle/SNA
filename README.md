# Stochastic Numerical Analysis (SNA) Experiment Framework

This framework is designed for running, comparing, and analyzing various stochastic optimization algorithms. It is primarily focused on second-order methods and is equipped with seamless [Weights & Biases (WandB)](https://wandb.ai/) integration for robust experiment tracking and visualization.

## Prerequisites

- Python 3.9+
- A Weights & Biases account for logging results online.

## Setup

1.  **Clone the Repository**
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
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

```
.
├── .completed_runs.log       # Tracks completed runs to avoid re-computation.
├── data/                     # Stores downloaded real-world datasets (e.g., MNIST).
├── optimizers/               # YAML configuration files for optimizers.
│   ├── base/
│   └── ...
├── problems/                 # YAML configuration files for problems (datasets/models).
│   ├── real/
│   └── synthetic/
├── datasets.py               # Data loading and synthetic data generation logic.
├── main.py                   # Main script for running experiments.
├── make_table.py             # Script to fetch results from WandB and generate tables.
├── objective_functions.py    # Definitions for loss functions (e.g., LinearRegression).
├── optimizers.py             # Implementations of optimization algorithms (SGD, SNA, mSNA).
├── run.py                    # Core experiment execution logic called by main.py.
├── utils.py                  # Utility functions for config loading, etc.
└── requirements.txt          # Python dependencies.
```

## How It Works

The framework operates based on a hierarchical configuration system using YAML files, separating the **problem** from the **optimizer**.

1.  **`main.py`** is the main entry point. It parses command-line arguments to find problem and optimizer configuration files.
2.  For each combination of problem and optimizer, it creates a unique configuration.
3.  It checks the `.completed_runs.log` to see if this exact configuration has already been run. If so, it skips it.
4.  It initializes a WandB run and calls **`run.py`** to execute the experiment for a specific seed.
5.  **`run.py`** sets up the dataset and model, runs the optimization loop, and logs all metrics to WandB.
6.  Once experiments are complete, **`make_table.py`** can be used to fetch the results from WandB and display them in a formatted table.

---

## Configuration Files

### 1. Problem Configuration

Defines the dataset and model. Located in `problems/`.

**Example (`problems/real/adult.yaml`):**
```yaml
problem_type: "real"
dataset: "adult"
model_params:
  name: "logistic_regression"
  bias: True
  lambda_: 0.001
dataset_params:
  test_size: 0.2
  val_size: 0.0
```

**Key Fields:**
- `problem_type`: (str) `real` or `synthetic`.
- `dataset`: (str) The name of the dataset. For `real` problems, this must match a loader in `datasets.py`. For `synthetic`, it defines the type (e.g., `synthetic_linear_regression`).
- `model_params`: (dict) Parameters for the objective function.
  - `name`: (str) The objective function to use (e.g., `logistic_regression`).
- `dataset_params`: (dict) Parameters for the data loader or generator.

### 2. Optimizer Configuration

Defines the algorithm and its hyperparameters. Located in `optimizers/`.

**Example (`optimizers/compare/mSNA.yaml`):**
```yaml
extends: mSNA_base.yaml
optimizer_params:
  lr_exp: 0.75
  lr_const: 1.0
  lr_add: 0.0
  averaged_matrix: True
  version: "mask"
  mask_size: 1
```

**Key Fields:**
- `extends`: (str, optional) Inherit from a base configuration in `optimizers/base/`.
- `optimizer`: (str) Name of the optimizer class (e.g., `SGD`, `mSNA`).
- `optimizer_params`: (dict) Hyperparameters for the optimizer instance.
- `initialization`: (dict, optional) See section on Optimizer Initialization.

---

## Running Experiments

Use `main.py` to launch one or more experiment runs.

**Arguments:**
- `-p, --problems`: Path(s) to problem YAML files. Wildcards are supported.
- `-o, --optimizers`: Path(s) to optimizer YAML files. Wildcards are supported.
- `-N, --num-seeds`: Number of seeds to run for each configuration.

**Examples:**
```bash
# Run mSNA on the adult dataset for 5 seeds
python main.py -p problems/real/adult.yaml -o optimizers/compare/mSNA.yaml -N 5

# Run all optimizers in 'compare' on all 'real' problems
python main.py -p 'problems/real/*' -o 'optimizers/compare/*' -N 3
```

## Generating Results Tables

After running experiments, use `make_table.py` to fetch results and generate a summary table. It uses the same arguments as `main.py` to identify which runs to include.

**Arguments:**
- `--skip-run`: **(Important)** Add this flag to prevent re-running experiments.
- All other arguments are the same as `main.py`.

**Example:**
```bash
# Fetch results for the runs specified and print a table
python make_table.py -p 'problems/real/*' -o 'optimizers/compare/*' -N 3 --skip-run
```

## Weights & Biases Integration

- **Setup**: Before the first run, open `main.py` and change the `WANDB_ENTITY` variable to your WandB username or team name.
- **Project Name**: The WandB project is named after the `dataset` field in the problem config (e.g., "adult", "mnist").
- **Group Name**: Runs are grouped by the optimizer's YAML filename (e.g., `mSNA`, `SGD_Avg`).
- **Run Name**: Each run has a unique ID based on its configuration and seed, ensuring reproducibility.

## Extending the Framework

### Adding a New Optimizer

1.  **Implement Logic**: Add a new class inheriting from `BaseOptimizer` in `optimizers.py`.
2.  **Register Class**: Add the new class to the `get_optimizer_class` function in `run.py`.
3.  **Create Config**: Create a new YAML file in the `optimizers/` directory. It's best practice to create a `my_optimizer_base.yaml` in `optimizers/base/` and have specific variants inherit from it.

### Adding a New Dataset

1.  **Implement Loader**: In `datasets.py`, add a new loader function (e.g., `load_my_dataset`) and add a corresponding case in the `load_dataset_from_source` function.
2.  **Create Config**: Create a problem YAML file in `problems/real/` that specifies the new `dataset` name.
