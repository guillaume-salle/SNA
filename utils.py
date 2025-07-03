import os
import yaml
import collections.abc
import math
import numpy as np
import glob
from typing import List, Dict, Any
from collections import defaultdict
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from torch.utils.data import DataLoader
from objective_functions import (
    BaseObjectiveFunction,
    LinearRegression,
    LogisticRegression,
)
from optimizers import SGD, mSNA, SNA, BaseOptimizer
import wandb
from datasets import load_dataset_from_source


def expand_file_patterns(patterns: List[str]) -> List[str]:
    """
    Expands a list of glob patterns into a list of unique, sorted file paths.
    """
    files = []
    for pattern in patterns:
        expanded_files = glob.glob(pattern, recursive=True)
        if not expanded_files:
            print(f"Warning: The pattern '{pattern}' did not match any files.")
        files.extend(expanded_files)
    # Return a sorted list of unique file paths
    return sorted(list(set(files)))


# ============================================================================ #
# >>> Configuration Utilities <<<                                              #
# ============================================================================ #


def deep_merge(source: dict, destination: dict) -> dict:
    """
    Deeply merges source dict into destination dict.
    Keys in source override keys in destination.
    Nested dictionaries are merged recursively.
    """
    for key, value in source.items():
        if isinstance(value, collections.abc.Mapping):
            # Get node or create one if it doesn't exist
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination


def evaluate_expression(expr: str, context: dict) -> float:
    """
    Safely evaluate a mathematical expression with variables from context.
    The context includes global variables like 'd' and local variables
    from the same config block.
    """
    # Create a safe environment with only allowed operations
    safe_dict = {
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "round": round,
        "sum": sum,
        "int": int,
        "float": float,
        "math": math,
    }
    # The context contains both global variables (like 'd') and
    # already-evaluated local variables from the config block.
    safe_dict.update(context)

    try:
        # Eval is used here, but with a carefully controlled and safe scope.
        return eval(expr, {"__builtins__": {}}, safe_dict)
    except NameError:
        # Re-raise NameError specifically. This allows the calling function
        # to catch it and handle dependency-based retries.
        raise
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {str(e)}")


def process_config_values(config: dict, context: dict) -> dict:
    """
    Process config values, evaluating expressions marked with 'expr:'.
    This function handles dependencies between expressions in the same block.
    """
    processed_config = {}
    unprocessed_expressions = {}

    # First pass: process non-expressions and identify all expressions
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            # Pass a copy of the context to avoid child contexts polluting parent contexts
            processed_config[key] = process_config_values(value, context.copy())
        elif isinstance(value, str) and value.startswith("expr:"):
            unprocessed_expressions[key] = value[5:].strip()
        else:
            processed_config[key] = value

    # Create a local context for expression evaluation within this block.
    # It combines the global context with the non-expression values from the current block.
    local_context = {**context, **processed_config}

    # Second pass: iteratively evaluate expressions, handling dependencies
    while unprocessed_expressions:
        processed_this_round = []
        for key, expr in unprocessed_expressions.items():
            try:
                # Attempt to evaluate the expression with the current local_context
                eval_result = evaluate_expression(expr, local_context)
                processed_config[key] = eval_result
                local_context[key] = eval_result  # Add newly evaluated value to the context
                processed_this_round.append(key)
            except NameError:
                # This expression depends on another one not yet processed.
                # We will retry in the next iteration.
                continue
            except Exception as e:
                # Handle other potential evaluation errors
                raise ValueError(f"Error processing expression for key '{key}': {e}")

        if not processed_this_round:
            # If we went through a whole round without processing anything,
            # it means there's a circular dependency.
            raise ValueError(
                f"Circular dependency or undefined variable in expressions: {list(unprocessed_expressions.keys())}"
            )

        # Remove the processed keys for the next iteration
        for key in processed_this_round:
            del unprocessed_expressions[key]

    return processed_config


def config_to_stable_string(cfg_item):
    """
    Converts a config (possibly nested dict/list/tuple) into a deterministic string representation.

    The "stable" in the name means that the output string is always the same for logically equivalent configs,
    regardless of the original key order in dictionaries. This is important for hashing or comparing configs,
    so that two configs with the same content but different key orders produce the same string.

    Args:
        cfg_item: The config item (dict, list, tuple, or primitive) to convert.

    Returns:
        str: A deterministic string representation of the config.
    """
    if isinstance(cfg_item, dict):
        # Sort keys to ensure deterministic order
        return "{" + ",".join(f"{k}:{config_to_stable_string(v)}" for k, v in sorted(cfg_item.items())) + "}"
    elif isinstance(cfg_item, list):
        return "[" + ",".join(config_to_stable_string(i) for i in cfg_item) + "]"
    elif isinstance(cfg_item, tuple):
        return "(" + ",".join(config_to_stable_string(i) for i in cfg_item) + ")"
    else:
        return str(cfg_item)


def sanitize_for_wandb(name: str) -> str:
    """Sanitizes a string for use as a wandb run/group/project name."""
    # The error messages from wandb suggest it URL-encodes names.
    # Characters like ^ , = are encoded and cause lookup failures.
    # We replace them with safe, descriptive alternatives.
    name = name.replace("^", "pow")
    name = name.replace("=", "_eq_")
    name = name.replace(",", "_")
    return name


def load_and_process_config(config_path: str, context: dict) -> dict:
    """
    Loads a YAML configuration file and processes any dynamic values.
    It supports inheriting from a base config using the 'extends' keyword.
    """
    config = load_config(config_path)

    # The context should be passed to the value processing function
    processed_config = process_config_values(config, context or {})
    return processed_config


def load_config(config_path, _load_chain=None):
    """
    Loads a single YAML configuration file, handling inheritance.
    It supports both 'extends' and 'inherits_from' for backward compatibility.
    """
    if _load_chain is None:
        _load_chain = []
    if config_path in _load_chain:
        raise ValueError(f"Circular dependency in config extension: {' -> '.join(_load_chain + [config_path])}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle inheritance, supporting both 'extends' and 'inherits_from'
    base_config_name = config.pop("extends", None)
    if base_config_name is None:
        base_config_name = config.pop("inherits_from", None)

    if base_config_name:
        # Ensure the base config name ends with .yaml if it doesn't already
        if not base_config_name.endswith(".yaml"):
            base_config_name += ".yaml"

        # First, search for the base config in the same directory as the child.
        current_config_dir = os.path.dirname(config_path)
        base_config_path = os.path.join(current_config_dir, base_config_name)

        # If not found locally, fall back to the central base directory.
        if not os.path.isfile(base_config_path):
            base_config_path = os.path.join("optimizers/base/", base_config_name)

        if not os.path.exists(base_config_path):
            raise FileNotFoundError(
                f"In config file '{config_path}', base config '{base_config_name}' not found. Searched in '{current_config_dir}' and a base directory."
            )

        base_config = load_config(base_config_path, _load_chain + [config_path])
        config = deep_merge(config, base_config)

    return config


# ============================================================================ #
# >>> Run Completion Manager <<<                                               #
# ============================================================================ #


class RunCompletionManager:
    """
    Manages the completion log file and cache for tracking completed runs.
    The log file is structured with project names as headers.
    A completed run is stored as `descriptive_name,wandb_id`.
    """

    DEFAULT_LOG_FILE = ".completed_runs.log"

    def __init__(self, log_filepath: str = DEFAULT_LOG_FILE):
        """
        Initializes the manager with the path to the completion log file.

        Args:
            log_filepath (str): The path to the log file.
        """
        self.log_filepath = log_filepath
        # Cache maps project name to a dict of {descriptive run name: wandb_id}
        self._completed_runs_cache: dict[str, dict[str, str]] | None = None
        print(f"RunCompletionManager initialized with log file: {self.log_filepath}")

    def get_log_data(self) -> dict[str, dict[str, str]]:
        """Reads the log file if not already cached and returns the cache."""
        if self._completed_runs_cache is None:
            self._read_log_file()
        return self._completed_runs_cache if self._completed_runs_cache is not None else {}

    def _read_log_file(self) -> None:
        """
        Reads the completion log file and populates the cache.
        The cache maps project names to a dictionary of {run_name: wandb_id}.
        """
        completed_runs_by_project = defaultdict(dict)
        current_project = None
        try:
            with open(self.log_filepath, "r") as f:
                for line in f:
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue

                    # Check for a project header (e.g., "my_project:")
                    if stripped_line.endswith(":") and not line.startswith(" "):
                        current_project = stripped_line[:-1]
                        continue

                    # Check for an indented run entry
                    if current_project and "," in stripped_line and (line.startswith(" ") or line.startswith("\t")):
                        parts = [p.strip() for p in stripped_line.split(",")]
                        if len(parts) == 3:
                            run_name, wandb_id, local_dir = parts
                            # Store both wandb_id and local_dir
                            completed_runs_by_project[current_project][run_name] = {
                                "wandb_id": wandb_id,
                                "local_dir": local_dir,
                            }
                        elif len(parts) == 2:  # Legacy support
                            run_name, wandb_id = parts
                            completed_runs_by_project[current_project][run_name] = {
                                "wandb_id": wandb_id,
                                "local_dir": None,
                            }

            if completed_runs_by_project or os.path.exists(self.log_filepath):
                total_runs = sum(len(p_runs) for p_runs in completed_runs_by_project.values())
                print(f"--> Read {total_runs} entries from completion log: {self.log_filepath}")

        except FileNotFoundError:
            print(f"--> Completion log file not found (normal for first run): {self.log_filepath}")
        except Exception as e:
            print(f"!!! Warning: Failed to read completion log file {self.log_filepath}: {e} !!!")
            self._completed_runs_cache = None  # Invalidate cache on error
            raise  # Re-raise the exception after logging
        self._completed_runs_cache = dict(completed_runs_by_project)

    def check_if_run_completed(self, expected_run_name: str) -> bool:
        """
        Checks if a run's descriptive name exists in the completion log cache.
        For backward compatibility, this checks across all projects.

        Args:
            expected_run_name: The unique descriptive identifier for the run.

        Returns:
            True if the run identifier is found in the log file, False otherwise.
        """
        if self._completed_runs_cache is None:
            self._read_log_file()
            if self._completed_runs_cache is None:
                print("!!! Warning: Cache is None after attempting read, assuming run not completed due to read error.")
                return False  # Cannot confirm completion if read failed

        # Check for the run name within any project
        for project_runs in self._completed_runs_cache.values():
            if expected_run_name in project_runs:
                return True
        return False

    def log_run_completion(self, run_name: str, wandb_id: str, local_dir: str, project_name: str) -> None:
        """
        Logs a completed run by mapping its descriptive name to its wandb ID and local directory.
        This method is not thread-safe but is sufficient for sequential runs.

        Args:
            run_name (str): The unique descriptive identifier of the completed run.
            wandb_id (str): The actual ID assigned by wandb.
            local_dir (str): The path to the local wandb run directory.
            project_name (str): The name of the project for grouping.
        """
        try:
            # Read all existing lines from the log file
            try:
                with open(self.log_filepath, "r") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                lines = []

            # Prepare the new entries
            project_header = f"{project_name}:"
            new_run_line = f"    {run_name},{wandb_id},{local_dir}\n"
            project_index = -1
            run_found_and_updated = False

            # Find the project block and try to update the run if it exists
            for i, line in enumerate(lines):
                if line.strip() == project_header:
                    project_index = i
                    # Search within the project block
                    for j in range(i + 1, len(lines)):
                        # Stop if we hit the next project or end of file
                        if not lines[j].strip() or not lines[j].startswith((" ", "\t")):
                            break
                        # Check if the run name matches at the start of the line
                        if lines[j].strip().startswith(run_name + ","):
                            lines[j] = new_run_line
                            run_found_and_updated = True
                            break
                    break  # Project found, no need to search further

            # If the run was not found and updated, it needs to be added.
            if not run_found_and_updated:
                if project_index != -1:
                    # Project exists, find where to insert the new run
                    insert_index = project_index + 1
                    while insert_index < len(lines) and lines[insert_index].startswith((" ", "\t")):
                        insert_index += 1
                    lines.insert(insert_index, new_run_line)
                else:
                    # Project doesn't exist, create it and add the run
                    if lines and not lines[-1].endswith(("\n", "\r")):
                        lines.append("\n")
                    lines.append(f"{project_header}\n")
                    lines.append(new_run_line)

            # Write the updated content back to the file
            with open(self.log_filepath, "w") as f:
                f.writelines(lines)

            log_action = "Updated" if run_found_and_updated else "Added"
            print(
                f"  [Completion Log] {log_action} run in log under project '{project_name}': {run_name} -> {wandb_id}"
            )

            # Update cache if it's already loaded
            if self._completed_runs_cache is not None:
                if project_name not in self._completed_runs_cache:
                    self._completed_runs_cache[project_name] = {}
                self._completed_runs_cache[project_name][run_name] = {"wandb_id": wandb_id, "local_dir": local_dir}
        except Exception as e:
            print(f"  [Completion Log] Warning: Failed to write to completion log {self.log_filepath}: {e}")
            # Invalidate cache if write fails, as its state might be inconsistent
            self._completed_runs_cache = None


# ============================================================================ #
# >>> Visualization Utilities <<<
# ============================================================================ #


def generate_dataset_characteristics_table(problem_configs: List[Dict[str, Any]]):
    """
    Generates a LaTeX table with characteristics of the datasets.
    """
    print(f"\n--- Generating LaTeX table for dataset characteristics... ---")

    table_data = []
    for p_config in problem_configs:
        dataset_name = p_config.get("dataset")
        dataset_params = p_config.get("dataset_params", {})

        # To get the exact train/test/init sizes, we need to load the dataset
        loaded_data = load_dataset_from_source(
            dataset_name=dataset_name,
            random_state=0,  # seed doesn't matter for sizes
            **dataset_params,
        )

        n_features = loaded_data["number_features"]
        n_train = loaded_data["n_train"]
        n_test = loaded_data["n_test"]
        init_size = dataset_params.get("init_size", 0)

        table_data.append(
            {
                "Dataset": dataset_name.capitalize(),
                "Features": n_features,
                "Training Set Size": n_train,
                "Init Set Size": init_size,
                "Testing Set Size": n_test,
            }
        )

    if not table_data:
        print("No real-data problem configs found to generate characteristics table.")
        return

    df = pd.DataFrame(table_data)
    df.set_index("Dataset", inplace=True)
    df.index.name = None  # Remove the index name to avoid the extra "Dataset" line in LaTeX

    def comma_formatter(x):
        return f"{x:,}"

    styler = df.style.format(
        {
            "Features": comma_formatter,
            "Training Set Size": comma_formatter,
            "Initialization Set Size": comma_formatter,
            "Testing Set Size": comma_formatter,
        }
    )
    latex_string = styler.to_latex(
        column_format="lrrrr",
        caption="Key characteristics of the datasets used in this study.",
        label="tab:dataset_characteristics",
        hrules=True,  # Use hrules for booktabs style
    )

    print("\n" + "=" * 50)
    print("COPY AND PASTE THE FOLLOWING LATEX CODE INTO YOUR .tex FILE")
    print("=" * 50 + "\n")
    print(latex_string)
    print("\n" + "=" * 50)
    print("Remember to include \\usepackage{booktabs} in your preamble.")
    print("=" * 50 + "\n")


def parse_local_run(local_dir: str, metrics_to_plot: List[str]) -> Dict | None:
    """
    Parses summary and history files from a local wandb run directory.
    It robustly fetches both final (scalar) metrics and time-series data.
    """
    if not local_dir or not os.path.isdir(local_dir):
        print(f"Warning: local directory not found or invalid: {local_dir}")
        return None

    # Accommodate paths that may or may not include the 'files' subdirectory.
    if os.path.basename(local_dir) == "files":
        files_dir = local_dir
    else:
        files_dir = os.path.join(local_dir, "files")

    summary_file = os.path.join(files_dir, "wandb-summary.json")
    history_file = os.path.join(files_dir, "wandb-history.jsonl")

    # 1. Parse the history file (.jsonl) to get time-series data and the last entry.
    metric_history = {metric: [] for metric in metrics_to_plot}
    last_history_item = {}
    try:
        with open(history_file, "r") as f:
            for line in f:
                try:
                    history_item = json.loads(line)
                    last_history_item = history_item  # Continuously update to get the last valid line
                except json.JSONDecodeError:
                    continue  # Skip corrupted lines

                # For metrics that need a full plot, collect their history.
                if "samples" in history_item:
                    for metric in metrics_to_plot:
                        if metric in history_item:
                            metric_history[metric].append(
                                {"samples": history_item["samples"], metric: history_item[metric]}
                            )
    except (FileNotFoundError, IOError):
        print(f"Warning: History file not found in {files_dir}. Only final metrics will be available.")

    # 2. Get final metric values. The summary file is the primary source.
    final_metrics = {}
    try:
        with open(summary_file, "r") as f:
            final_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        # If the summary file is missing or corrupt, use the last history line as a fallback.
        final_metrics = last_history_item
        if final_metrics:
            print(f"Warning: Summary file not found in {files_dir}. Using last history line for final metrics.")

    # 3. Assemble the final run_data dictionary.
    # Start with the final metrics.
    run_data: Dict[str, Any] = {**final_metrics}
    # Add the history data under the "history" key.
    run_data["history"] = {metric: data for metric, data in metric_history.items() if data}

    return run_data


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Pivot the table to get optimizers as columns and metrics as rows
    pivot_df = df.pivot_table(index=["Dataset", "Metric"], columns="Optimizer", values="Value")

    # Define a formatter for the time values
    def format_time(seconds):
        if pd.isna(seconds):
            return ""
        if seconds < 1.0:
            return f"{seconds * 1000:.1f} ms"
        else:
            return f"{seconds:.2f} s"

    # Create a new DataFrame for formatted strings to avoid warnings
    formatted_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns, dtype=object)

    for idx, row in pivot_df.iterrows():
        metric_name = idx[1]

        # Determine the best value and its column name
        if metric_name in ["Train Acc", "Test Acc"]:
            best_val = row.max()
        else:  # For Time and Loss, lower is better
            best_val = row.min()

        for col_name, value in row.items():
            is_best = value == best_val
            prefix = "* " if is_best else "  "

            # Apply formatting based on metric type
            if metric_name == "Time":
                formatted_value = format_time(value)
            elif "Acc" in metric_name:
                formatted_value = f"{value:.2f}"
            elif "Loss" in metric_name:
                formatted_value = f"{value:.2e}"
            else:
                formatted_value = str(value)

            formatted_df.loc[idx, col_name] = prefix + formatted_value

    # Reorder the metrics to a logical sequence
    metric_order = ["Train Acc", "Test Acc", "Train Loss", "Test Loss", "Time"]
    formatted_df = formatted_df.reindex(metric_order, level="Metric")

    # Clean up the index names for presentation
    formatted_df.index.names = ["Dataset", ""]

    return formatted_df


def generate_accuracy_table(runs: List[Dict[str, Any]], latex: bool = False):
    print(f"\n--- Generating {'LaTeX' if latex else 'accuracy'} table from local run data... ---")
    results_by_config = defaultdict(
        lambda: defaultdict(
            lambda: {
                "train_accs": [],
                "test_accs": [],
                "optimizer_times": [],
                "train_losses": [],
                "test_losses": [],
            }
        )
    )

    for run_info in runs:
        p_name, o_name = run_info["p_name"], run_info["o_name"]
        local_data = parse_local_run(run_info["local_dir"], [])
        if not local_data:
            print(f"Warning: Could not parse local data for run {run_info['wandb_id']}. Skipping.")
            continue

        required_metrics = [
            "final_train_accuracy",
            "final_test_accuracy",
            "optimizer_time",
            "final_train_loss",
            "final_test_loss",
        ]
        if all(key in local_data for key in required_metrics):
            results_by_config[p_name][o_name]["train_accs"].append(local_data["final_train_accuracy"])
            results_by_config[p_name][o_name]["test_accs"].append(local_data["final_test_accuracy"])
            results_by_config[p_name][o_name]["optimizer_times"].append(local_data["optimizer_time"])
            results_by_config[p_name][o_name]["train_losses"].append(local_data["final_train_loss"])
            results_by_config[p_name][o_name]["test_losses"].append(local_data["final_test_loss"])
        else:
            # Create a more informative warning message
            missing_keys = [key for key in required_metrics if key not in local_data]
            print(
                f"Warning: Missing required metrics {missing_keys} in local files for run {run_info['wandb_id']}. Skipping."
            )

    all_results = []
    for p_name, optimizers in results_by_config.items():
        for o_name, results in optimizers.items():
            if all(
                results.get(key)
                for key in ["train_accs", "test_accs", "optimizer_times", "train_losses", "test_losses"]
            ):
                avg_train_acc = sum(results["train_accs"]) / len(results["train_accs"])
                avg_test_acc = sum(results["test_accs"]) / len(results["test_accs"])
                avg_time = sum(results["optimizer_times"]) / len(results["optimizer_times"])
                avg_train_loss = sum(results["train_losses"]) / len(results["train_losses"])
                avg_test_loss = sum(results["test_losses"]) / len(results["test_losses"])

                if latex:
                    # For LaTeX, create a different data structure
                    all_results.append(
                        {
                            "Dataset": p_name,
                            "Optimizer": o_name,
                            "Train Acc": avg_train_acc * 100,
                            "Test Acc": avg_test_acc * 100,
                            "Train Loss": avg_train_loss,
                            "Test Loss": avg_test_loss,
                            "Time": avg_time,
                        }
                    )
                else:
                    # For console table, use the original structure
                    all_results.append(
                        {"Dataset": p_name, "Metric": "Train Acc", "Optimizer": o_name, "Value": avg_train_acc * 100}
                    )
                    all_results.append(
                        {"Dataset": p_name, "Metric": "Test Acc", "Optimizer": o_name, "Value": avg_test_acc * 100}
                    )
                    all_results.append(
                        {"Dataset": p_name, "Metric": "Train Loss", "Optimizer": o_name, "Value": avg_train_loss}
                    )
                    all_results.append(
                        {"Dataset": p_name, "Metric": "Test Loss", "Optimizer": o_name, "Value": avg_test_loss}
                    )
                    all_results.append({"Dataset": p_name, "Metric": "Time", "Optimizer": o_name, "Value": avg_time})

    if not all_results:
        print("No results to display.")
        return

    if latex:
        # Generate LaTeX table
        _generate_latex_table(pd.DataFrame(all_results))
    else:
        # Generate console table
        final_table = format_table(pd.DataFrame(all_results))
        print("\n--- Results Table ---")
        print(final_table.to_string())


def _generate_latex_table(df: pd.DataFrame):
    """
    Helper function to generate a publication-quality LaTeX table from DataFrame.
    """
    # Rename optimizers first for cleaner, more readable names in the table
    optimizer_rename = {
        "Stream_SGD": "SGD",
        "Stream_SGD_Avg": "SGD-Avg",
        "Stream_mSNA": "mSNA",
        "Stream_mSNA_Avg": "mSNA-Avg",
        "Stream_mSNA_ell-0,25": "mSNA (l=0.25)",
        "Stream_mSNA_ell-0,5": "mSNA (l=0.5)",
        "Stream_mSNA_init_hess-10d": "mSNA",
        "Stream_mSNA_Avg_init_hess-10d": "mSNA-Avg",
    }
    df["Optimizer"] = df["Optimizer"].replace(optimizer_rename)

    # **IMPORTANT**: Escape LaTeX special characters AFTER renaming.
    # This ensures that underscores in original names don't prevent remapping.
    df["Optimizer"] = df["Optimizer"].str.replace("_", r"\_", regex=False)
    df["Dataset"] = df["Dataset"].str.replace("_", r"\_", regex=False)

    # Set up the multi-index which is key for grouping and multirow
    df = df.set_index(["Dataset", "Optimizer"])

    # Create an empty DataFrame to hold style information ('font-weight: bold')
    style_df = pd.DataFrame("", index=df.index, columns=df.columns)

    metrics_higher_better = ["Train Acc", "Test Acc"]

    # Iterate over each dataset to find and mark the best value
    for dataset_name in df.index.get_level_values("Dataset").unique():
        sub_df = df.loc[dataset_name]
        for col in df.columns:
            if col in metrics_higher_better:
                best_optimizer_idx = sub_df[col].idxmax()
            else:  # Lower is better for Loss and Time
                best_optimizer_idx = sub_df[col].idxmin()

            # Mark the cell for bolding
            style_df.loc[(dataset_name, best_optimizer_idx), col] = "font-weight: bold"

    # Use the Styler to apply formatting and styles
    styler = df.style.apply(lambda s: style_df, axis=None)

    def format_time(seconds):
        if pd.isna(seconds):
            return ""
        return f"{seconds * 1000:.1f} ms" if seconds < 1.0 else f"{seconds:.2f} s"

    styler = styler.format(
        {
            "Train Acc": "{:.2f}",
            "Test Acc": "{:.2f}",
            "Train Loss": "{:.2e}",
            "Test Loss": "{:.2e}",
            "Time": format_time,
        }
    )

    # --- Generate LaTeX Code ---
    latex_string = styler.to_latex(
        column_format="llrrrrr",
        position="!htbp",
        caption="Performance of streaming optimizers on various datasets.",
        label="tab:performance_results",
        hrules=True,  # Key for booktabs and automatic multirow grouping
        convert_css=True,  # Key to convert 'font-weight: bold' to \textbf{}
    )

    # Manually insert \midrule between dataset groups for better readability
    lines = latex_string.splitlines()
    new_lines = []
    # Find the index of the header's bottom rule
    try:
        midrule_index = [i for i, s in enumerate(lines) if r"\midrule" in s][0]
    except IndexError:
        midrule_index = -1

    if midrule_index != -1:
        # Add lines up to and including the header's rule
        new_lines.extend(lines[: midrule_index + 1])
        first_multirow_found = False
        # Process the data rows
        for line in lines[midrule_index + 1 :]:
            # A new dataset block is starting if the line contains \multirow
            if r"\multirow" in line:
                # Add a \midrule before each new block, except the very first one
                if first_multirow_found:
                    new_lines.append(r"\midrule")
                else:
                    first_multirow_found = True
            new_lines.append(line)
        latex_string = "\n".join(new_lines)

    print("\n" + "=" * 50)
    print("COPY AND PASTE THE FOLLOWING LATEX CODE INTO YOUR .tex FILE")
    print("=" * 50 + "\n")
    print(latex_string)
    print("\n" + "=" * 50)
    print("Remember to include \\usepackage{booktabs} and \\usepackage{multirow} in your preamble.")
    print("=" * 50 + "\n")


def generate_combined_synthetic_plot(runs: List[Dict[str, Any]], entity: str, color_map: Dict[str, Any] | None = None):
    """
    Generates a single figure with three subplots for synthetic data:
    1. Estimation Error (line)
    2. Inverse Hessian Error (line)
    3. Optimizer Time (bar)
    Includes a common legend at the bottom.
    """
    if not entity:
        print("!!! ERROR: W&B entity not provided. Cannot fetch data from API. !!!")
        return

    print("\n--- Generating combined plot for synthetic datasets... ---")

    # Group runs by project name (p_name)
    runs_by_project = defaultdict(list)
    for run_info in runs:
        runs_by_project[run_info["p_name"]].append(run_info)

    api = wandb.Api()

    for p_name, project_runs in runs_by_project.items():
        # Get the first run's config to extract the dimension
        if project_runs:
            first_run = project_runs[0]
            run_path = f"{entity}/{p_name}/{first_run['wandb_id']}"
            run = api.run(run_path)
            if "dataset_params" in run.config and "param_dim" in run.config["dataset_params"]:
                d = run.config["dataset_params"]["param_dim"]
                print(f"   Using dimension d={d} for project {p_name}")
            else:
                raise ValueError(f"Could not find param_dim in config for project {p_name}")

        fig, axes = plt.subplots(
            1, 3, figsize=(12, 3.5), gridspec_kw={"width_ratios": [1, 1, 0.6]}
        )  # Make bar plot narrower

        # Group runs by optimizer for easier iteration
        runs_by_optimizer = defaultdict(list)
        for run_info in project_runs:
            runs_by_optimizer[run_info["o_name"]].append(run_info)

        # Define preferred ordering for optimizers
        optimizer_order_preference = [
            "SGD",
            "SGD-Avg",
            r"mSNA ($\ell=1$)",
            r"mSNA-Avg ($\ell=1$)",
            r"mSNA ($\ell=d^{0.25}$)",
            r"mSNA-Avg ($\ell=d^{0.25}$)",
            r"mSNA ($\ell=d^{0.5}$)",
            r"mSNA-Avg ($\ell=d^{0.5}$)",
        ]

        # Create ordered optimizer names based on preference and what's actually present
        all_optimizer_names = set(runs_by_optimizer.keys())
        ordered_optimizer_names = []

        # First, add optimizers in preferred order if they exist
        for preferred_name in optimizer_order_preference:
            if preferred_name in all_optimizer_names:
                ordered_optimizer_names.append(preferred_name)

        # Then add any remaining optimizers not in the preference list
        for name in sorted(all_optimizer_names):
            if name not in optimizer_order_preference:
                ordered_optimizer_names.append(name)

        # --- PLOTS 1 & 2: Line plots for errors ---
        metrics_to_plot = ["estimation_error", "inv_hess_error_fro"]
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            for o_name in ordered_optimizer_names:
                run_list = runs_by_optimizer[o_name]
                all_histories = []
                for run_info in run_list:
                    try:
                        run_path = f"{entity}/{p_name}/{run_info['wandb_id']}"
                        run = api.run(run_path)
                        scanned_data = run.scan_history(keys=["samples", metric])
                        history = pd.DataFrame(list(scanned_data))
                        if not history.empty and metric in history.columns and not history[metric].dropna().empty:
                            all_histories.append(history.set_index("samples")[[metric]].dropna())
                    except Exception as e:
                        print(f"    -> WARNING: Could not fetch run {run_info['wandb_id']}. Error: {e}")

                if all_histories:
                    combined_history = pd.concat(all_histories)
                    mean_history = combined_history.groupby(combined_history.index).mean()
                    plot_color = color_map.get(o_name) if color_map else None
                    ax.plot(mean_history.index, mean_history[metric], label=o_name, color=plot_color)

            metric_name_map = {
                "estimation_error": r"$\|\theta_n - \theta^*\|^2$",
                "inv_hess_error_fro": r"$\|A_n - H^{-1}\|_F^2$",
            }
            ax.set_ylabel(metric_name_map.get(metric, metric), rotation="vertical")
            ax.set_xlabel("Samples")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=d)  # Start x-axis at d samples
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # --- PLOT 3: Optimizer Time ---
        ax = axes[2]
        optimizer_times = defaultdict(list)
        for o_name, run_list in runs_by_optimizer.items():
            for run_info in run_list:
                try:
                    run_path = f"{entity}/{p_name}/{run_info['wandb_id']}"
                    run = api.run(run_path)
                    if "optimizer_time" in run.summary:
                        optimizer_times[o_name].append(run.summary["optimizer_time"])
                except Exception as e:
                    print(f"    -> WARNING: Could not fetch run summary for {run_info['wandb_id']}. Error: {e}")

        if optimizer_times:
            avg_times = {o_name: sum(times) / len(times) for o_name, times in optimizer_times.items()}
            ordered_times = [avg_times[o_name] for o_name in ordered_optimizer_names]
            bar_colors = [color_map.get(o_name) for o_name in ordered_optimizer_names] if color_map else None
            ax.bar(ordered_optimizer_names, ordered_times, color=bar_colors, width=0.4)
            ax.set_xlabel("Optimizer")
            ax.set_ylabel("Optimizer Time (s)")
            ax.set_xticklabels([])  # Remove x-axis labels since they're in the legend

        # --- COMMON LEGEND & TITLE ---
        handles, labels = axes[0].get_legend_handles_labels()

        # Since plots were created in the correct order, the handles and labels should already be ordered correctly
        # Just use them as-is since they match the ordered_optimizer_names order
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(labels))

        fig.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout for legend

        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = os.path.join(plot_dir, f"combined_{p_name}.png")
        plt.savefig(plot_filename, bbox_inches="tight")
        plt.close()
        print(f"\nCombined plot saved to '{plot_filename}'")


def run_visualizations(
    runs_to_fetch: List[Dict[str, Any]],
    problem_files: List[str],
    problem_configs: Dict[str, Dict[str, Any]],
    latex: bool = False,
    entity: str | None = None,
):
    """
    Separates synthetic and real-data runs and generates the appropriate
    visualizations for each type (plots for synthetic, tables for real).
    """
    metrics_to_plot = ["estimation_error", "inv_hess_error_fro"]

    synthetic_runs = []
    real_runs = []

    # Create a mapping from problem name to whether it's synthetic based on file paths
    is_synthetic_map = {os.path.basename(p).replace(".yaml", ""): "synthetic" in p for p in problem_files}

    # Separate runs into synthetic and real based on the problem they belong to
    for run in runs_to_fetch:
        if is_synthetic_map.get(run["p_name"], False):
            synthetic_runs.append(run)
        else:
            real_runs.append(run)

    # Generate plots for any synthetic runs found
    if synthetic_runs:
        print("\n--- Generating plots for synthetic datasets... ---")

        # Define a rename mapping for cleaner optimizer names in plots
        optimizer_rename = {
            "Stream_SGD": "SGD",
            "Stream_SGD_Avg": "SGD-Avg",
            "Stream_mSNA": r"mSNA ($\ell=1$)",
            "Stream_mSNA_Avg": r"mSNA-Avg ($\ell=1$)",
            "Stream_mSNA_ell-0,25": r"mSNA ($\ell=d^{0.25}$)",
            "Stream_mSNA_Avg_ell-0,25": r"mSNA-Avg ($\ell=d^{0.25}$)",
            "Stream_mSNA_ell-0,5": r"mSNA ($\ell=d^{0.5}$)",
            "Stream_mSNA_Avg_ell-0,5": r"mSNA-Avg ($\ell=d^{0.5}$)",
            "Stream_mSNA_init_hess-10d": r"mSNA ($\ell=1$)",
            "Stream_mSNA_Avg_init_hess-10d": r"mSNA-Avg ($\ell=1$)",
        }
        # Apply the renaming to the synthetic runs data before plotting
        for run in synthetic_runs:
            run["o_name"] = optimizer_rename.get(run["o_name"], run["o_name"])

        # Create a consistent color map for all optimizers across all synthetic plots
        all_optimizer_names = sorted(list(set(run["o_name"] for run in synthetic_runs)))
        # Use a built-in colormap from matplotlib
        colormap = plt.get_cmap("tab10")
        color_map = {name: colormap(i) for i, name in enumerate(all_optimizer_names)}

        generate_combined_synthetic_plot(synthetic_runs, entity, color_map=color_map)

    # Generate tables for any real-data runs found
    if real_runs:
        print("\n--- Generating table for real datasets... ---")
        if latex:
            # Generate dataset characteristics table
            real_problem_configs = []
            for p_name, p_config in problem_configs.items():
                if not is_synthetic_map.get(p_name, False):
                    real_problem_configs.append(p_config)

            if real_problem_configs:
                generate_dataset_characteristics_table(real_problem_configs)

        generate_accuracy_table(real_runs, latex=latex)


# ============================================================================ #
# >>> Evaluation Utilities <<<
# ============================================================================ #


def evaluate_on_set(
    eval_set: torch.utils.data.Dataset,
    model: BaseObjectiveFunction,
    param: torch.Tensor,
    device: str,
    model_name: str,
    set_name: str,
    eval_batch_size: int = 512,
    subset_size: int | None = None,
) -> dict:
    """Helper to evaluate loss and accuracy on a given dataset or a subset of it."""
    if eval_set is None or len(eval_set) == 0:
        return {}

    sampler = None
    if subset_size and subset_size < len(eval_set):
        # Use a random subset of the data for evaluation
        indices = torch.randperm(len(eval_set))[:subset_size]
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        total_samples = subset_size
    else:
        total_samples = len(eval_set)

    eval_loader = DataLoader(eval_set, batch_size=eval_batch_size, pin_memory=(device == "cuda"), sampler=sampler)
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch_cpu, y_batch_cpu in eval_loader:
            X_batch, y_batch = X_batch_cpu.to(device), y_batch_cpu.to(device)
            loss = model((X_batch, y_batch), param)
            total_loss += loss.item() * X_batch.size(0)

            if model_name.lower() == "logistic_regression":
                phi_batch = model._add_bias(X_batch)
                logits = torch.matmul(phi_batch, param)
                predictions = (torch.sigmoid(logits) > 0.5).float()
                all_predictions.append(predictions.cpu())
                all_targets.append(y_batch_cpu.squeeze())

    metrics = {}
    if total_samples > 0:
        metrics[f"{set_name}_loss"] = total_loss / total_samples
        if all_predictions:
            predictions_tensor = torch.cat(all_predictions)
            targets_tensor = torch.cat(all_targets)
            correct_predictions = (predictions_tensor == targets_tensor).sum().item()
            metrics[f"{set_name}_accuracy"] = correct_predictions / total_samples
    return metrics


# ============================================================================ #
# >>> Class Getters <<<
# ============================================================================ #


def get_optimizer_class(optimizer_name: str) -> type[BaseOptimizer]:
    if optimizer_name == "SGD":
        return SGD
    elif optimizer_name == "mSNA":
        return mSNA
    elif optimizer_name == "SNA":
        return SNA
    else:
        raise ValueError(f"Unknown optimizer specified in config: {optimizer_name}")


def get_obj_function_class(model_type: str) -> Any:
    model_type_lower = model_type.lower()
    if model_type_lower == "linear_regression":
        return LinearRegression
    elif model_type_lower == "logistic_regression":
        return LogisticRegression
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================ #
# >>> Learning Rate Finder Utilities <<<
# ============================================================================ #


def find_best_lr(problem_config: Dict, seed: int):
    """
    Performs a line search to find an optimal initial learning rate.
    It tests a range of learning rates on a single large batch of data,
    computes the loss for one step, and plots the results.
    """
    print("--- Starting Learning Rate Finder ---")

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = problem_config.get("dataset")
    dataset_params = problem_config.get("dataset_params")
    model_name = problem_config.get("model")
    model_params = problem_config.get("model_params")

    model = get_obj_function_class(model_name)(**model_params)

    # --- Load Data ---
    # Load the full training set to select a large batch from it.
    print(f"Loading dataset '{dataset_name}' to create a search batch...")
    torch.manual_seed(seed)
    loaded_data = load_dataset_from_source(dataset_name=dataset_name, random_state=seed, **dataset_params)
    train_set = loaded_data["train_dataset"]
    number_features = loaded_data["number_features"]

    if len(train_set) == 0:
        print("Error: Training set is empty. Cannot run learning rate finder.")
        return

    # Create a single large batch for the search.
    search_batch_size = min(5000, len(train_set))
    data_loader = DataLoader(train_set, batch_size=search_batch_size, shuffle=True)
    search_data_cpu = next(iter(data_loader))
    search_data = tuple(item.to(device) for item in search_data_cpu)

    # --- Initialize Parameters ---
    bias = problem_config.get("model_params", {}).get("bias", False)
    param_dim = number_features + 1 if bias else number_features
    theta_init = torch.zeros(param_dim, device=device, dtype=torch.float32)

    # --- LR Search Loop ---
    # Test 100 learning rates logarithmically spaced from 1e-12 to 1000.
    lr_candidates = np.logspace(-12, 3, 100)
    search_steps_options = [10, 100, 1000]
    all_losses = {steps: [] for steps in search_steps_options}

    print(f"Testing {len(lr_candidates)} learning rates over {search_steps_options} steps...")

    for num_search_steps in search_steps_options:
        print(f"  Running search for {num_search_steps} steps...")
        losses_for_this_run = []
        for lr in lr_candidates:
            temp_theta = theta_init.clone()
            try:
                for _ in range(num_search_steps):
                    grad = model.grad(search_data, temp_theta)
                    if not torch.all(torch.isfinite(grad)):
                        raise RuntimeError("Gradient became NaN or Inf.")
                    temp_theta.add_(grad, alpha=-lr)
                loss = model(search_data, temp_theta)
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss became NaN or Inf.")
                losses_for_this_run.append(loss.item())
            except RuntimeError:
                losses_for_this_run.append(float("nan"))
        all_losses[num_search_steps] = losses_for_this_run

    # --- Plotting Results ---
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(search_steps_options)))
    best_lrs_info = {}

    for i, (num_steps, losses) in enumerate(all_losses.items()):
        losses_arr = np.array(losses)
        # Filter out NaNs for plotting to avoid gaps if library doesn't handle it
        finite_mask = np.isfinite(losses_arr)
        plt.plot(
            lr_candidates[finite_mask], losses_arr[finite_mask], label=f"Loss after {num_steps} steps", color=colors[i]
        )

        # Use nanargmin to find the best LR, ignoring NaNs.
        try:
            best_lr_index = np.nanargmin(losses_arr)
            best_lr = lr_candidates[best_lr_index]
            min_loss = losses_arr[best_lr_index]
            best_lrs_info[num_steps] = (best_lr, min_loss)

            plt.axvline(x=best_lr, color=colors[i], linestyle="--", label=f"Best for {num_steps} steps: {best_lr:.2e}")
        except ValueError:  # This happens if all losses are NaN
            print(f"Warning: All losses were NaN for {num_steps} steps. Could not find a best LR.")

    # Find the overall minimum loss to set the y-axis scale appropriately
    all_finite_losses = [loss for losses in all_losses.values() for loss in losses if np.isfinite(loss)]
    if all_finite_losses:
        global_min_loss = min(all_finite_losses)
        # Set a dynamic y-limit to zoom in on the interesting part of the plot
        # The upper limit is 10x the minimum loss, with a small buffer.
        # The lower limit is slightly below the minimum loss to provide some space.
        y_upper_limit = global_min_loss * 10 + 1
        y_lower_limit = global_min_loss - (abs(global_min_loss) * 0.1 + 0.1)
        plt.ylim(y_lower_limit, y_upper_limit)

    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title(f"Learning Rate Finder for {dataset_name.capitalize()}")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"lr_finder_{sanitize_for_wandb(dataset_name)}.png")
    plt.savefig(plot_path)

    print("\n--- Results ---")
    for num_steps, (best_lr, min_loss) in best_lrs_info.items():
        print(f"Optimal learning rate after {num_steps} steps: {best_lr:.4e} (Loss: {min_loss:.4f})")

    print(f"Plot saved to: {plot_path}")
    print("\nInspect the plot to confirm the choice. A good LR is typically the lowest point before the loss explodes.")
    print(
        "You can now update your optimizer's .yaml file. For a constant LR, set 'lr_const' to this value, 'lr_exp: 0.0', and 'lr_add: 0.0'."
    )
