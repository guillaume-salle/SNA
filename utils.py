import os
import yaml
import collections.abc
import math
import hashlib
import glob
from typing import List


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
    if isinstance(cfg_item, dict):
        # Sort keys, recursively process values
        return "{" + ",".join(f"{k}:{config_to_stable_string(v)}" for k, v in sorted(cfg_item.items())) + "}"
    elif isinstance(cfg_item, list):
        # Process list items recursively
        return "[" + ",".join(config_to_stable_string(i) for i in cfg_item) + "]"
    elif isinstance(cfg_item, tuple):
        # Process tuple items recursively
        return "(" + ",".join(config_to_stable_string(i) for i in cfg_item) + ")"
    else:
        # Convert other types to string
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
    processed_config = process_config_values(config, context)
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
        # Cache maps descriptive run name to wandb_id
        self._completed_runs_cache: dict[str, str] | None = None
        print(f"RunCompletionManager initialized with log file: {self.log_filepath}")

    def _read_log_file(self) -> None:
        """
        Reads the completion log file and populates the cache.
        The cache maps the descriptive run name to its wandb ID.
        """
        completed_runs = {}
        try:
            with open(self.log_filepath, "r") as f:
                for line in f:
                    # A run is an indented line containing a comma
                    if (
                        line.strip()
                        and not line.endswith(":")
                        and "," in line
                        and (line.startswith(" ") or line.startswith("\t"))
                    ):
                        parts = line.strip().split(",")
                        if len(parts) == 2:
                            run_name, wandb_id = parts
                            completed_runs[run_name] = wandb_id
            # Only print if file was actually read and had content potentially
            if completed_runs or os.path.exists(self.log_filepath):
                print(f"--> Read {len(completed_runs)} entries from completion log: {self.log_filepath}")
        except FileNotFoundError:
            print(f"--> Completion log file not found (normal for first run): {self.log_filepath}")
            # File doesn't exist yet, cache is an empty set
        except Exception as e:
            print(f"!!! Warning: Failed to read completion log file {self.log_filepath}: {e} !!!")
            # In case of error, don't trust potentially partial cache
            self._completed_runs_cache = None  # Invalidate cache on error
            raise  # Re-raise the exception after logging
        self._completed_runs_cache = completed_runs

    def check_if_run_completed(self, expected_run_name: str) -> bool:
        """
        Checks if a run's descriptive name exists in the completion log cache.

        Args:
            expected_run_name: The unique descriptive identifier for the run.

        Returns:
            True if the run identifier is found in the log file, False otherwise.
        """
        if self._completed_runs_cache is None:
            self._read_log_file()
            # _read_log_file sets the cache, handle potential None if error occurred during read
            if self._completed_runs_cache is None:
                print("!!! Warning: Cache is None after attempting read, assuming run not completed due to read error.")
                return False  # Cannot confirm completion if read failed

        return expected_run_name in self._completed_runs_cache

    def log_run_completion(self, run_name: str, wandb_id: str, project_name: str) -> None:
        """
        Logs a completed run by mapping its descriptive name to its wandb ID.
        This method is not thread-safe but is sufficient for sequential runs.

        Args:
            run_name (str): The unique descriptive identifier of the completed run.
            wandb_id (str): The actual ID assigned by wandb.
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
            # New line format includes the wandb ID
            new_run_line = f"    {run_name},{wandb_id}\n"

            project_index = -1
            for i, line in enumerate(lines):
                if line.strip() == project_header:
                    project_index = i
                    break

            if project_index != -1:
                # Find where to insert the new run within the project block
                insert_index = project_index + 1
                while insert_index < len(lines) and (
                    lines[insert_index].startswith(" ") or lines[insert_index].startswith("\t")
                ):
                    insert_index += 1
                lines.insert(insert_index, new_run_line)
            else:  # Project header not found, so add it
                # To keep it clean, add a newline before a new project if the file is not empty
                if lines and not lines[-1].endswith("\n"):
                    lines.append("\n")
                lines.append(f"{project_header}\n")
                lines.append(new_run_line)

            # Write the updated content back to the file
            with open(self.log_filepath, "w") as f:
                f.writelines(lines)

            print(f"  [Completion Log] Added run to log under project '{project_name}': {run_name} -> {wandb_id}")

            # Update cache if it's already loaded
            if self._completed_runs_cache is not None:
                self._completed_runs_cache[run_name] = wandb_id
        except Exception as e:
            print(f"  [Completion Log] Warning: Failed to write to completion log {self.log_filepath}: {e}")
            # Invalidate cache if write fails, as its state might be inconsistent
            self._completed_runs_cache = None
