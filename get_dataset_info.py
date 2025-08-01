import argparse
import os
import glob
from typing import List, Dict, Any
import yaml
import collections.abc
import math

# Import the original, robust data loading function
from datasets import load_dataset_from_source


# ============================================================================ #
# >>> Config Evaluation Utilities (Copied from utils.py for lightweight script)
# ============================================================================ #


def deep_merge(source: dict, destination: dict) -> dict:
    """Deeply merges source dict into destination dict."""
    for key, value in source.items():
        if isinstance(value, collections.abc.Mapping):
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination


def evaluate_expression(expr: str, context: dict) -> float:
    """Safely evaluate a mathematical expression with variables from context."""
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
    safe_dict.update(context)
    try:
        return eval(expr, {"__builtins__": {}}, safe_dict)
    except NameError:
        raise
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {str(e)}")


def process_config_values(config: dict, context: dict) -> dict:
    """Process config values by evaluating expressions marked with 'expr:'."""
    processed_config = {}
    unprocessed_expressions = {}

    for key, value in config.items():
        if isinstance(value, dict):
            processed_config[key] = process_config_values(value, context.copy())
        elif isinstance(value, str) and value.startswith("expr:"):
            unprocessed_expressions[key] = value[5:].strip()
        else:
            processed_config[key] = value

    local_context = {**context, **processed_config}

    while unprocessed_expressions:
        processed_this_round = []
        for key, expr in unprocessed_expressions.items():
            try:
                eval_result = evaluate_expression(expr, local_context)
                processed_config[key] = eval_result
                local_context[key] = eval_result
                processed_this_round.append(key)
            except NameError:
                continue
        if not processed_this_round:
            raise ValueError(
                f"Circular dependency or undefined variable in expressions: {list(unprocessed_expressions.keys())}"
            )
        for key in processed_this_round:
            del unprocessed_expressions[key]

    return processed_config


def expand_file_patterns(patterns: List[str]) -> List[str]:
    """Expands a list of glob patterns into a list of unique, sorted file paths."""
    files = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "**/*.yaml")
        expanded_files = glob.glob(pattern, recursive=True)
        if not expanded_files:
            print(f"Warning: The pattern '{pattern}' did not match any files.")
        files.extend(expanded_files)
    return sorted(list(set(f for f in files if os.path.isfile(f))))


def load_config_with_inheritance(config_path, _load_chain=None):
    """Loads a YAML file, handling 'extends'/'inherits_from' keywords."""
    if _load_chain is None:
        _load_chain = []
    if config_path in _load_chain:
        raise ValueError(f"Circular dependency in config extension: {' -> '.join(_load_chain + [config_path])}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_config_name = config.pop("extends", None) or config.pop("inherits_from", None)
    if base_config_name:
        if not base_config_name.endswith(".yaml"):
            base_config_name += ".yaml"
        current_dir = os.path.dirname(config_path)
        base_path = os.path.join(current_dir, base_config_name)
        if not os.path.isfile(base_path):
            base_path = os.path.join("optimizers/base/", base_config_name)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"In '{config_path}', base config '{base_config_name}' not found.")
        base_config = load_config_with_inheritance(base_path, _load_chain + [config_path])
        config = deep_merge(config, base_config)
    return config


# ============================================================================ #
# >>> Main Execution <<<
# ============================================================================ #


def main():
    """Parses arguments and prints dataset characteristics."""
    parser = argparse.ArgumentParser(description="Get information about real datasets.")
    parser.add_argument(
        "-p", "--problems", nargs="+", required=True, help="Path(s) to YAML problem configuration files."
    )
    args = parser.parse_args()

    problem_files = expand_file_patterns(args.problems)
    if not problem_files:
        print("No problem configuration files found.")
        return

    print("-" * 100)
    print(
        f"{'Dataset':<20} | {'Train Size':>12} | {'Init Size':>12} | {'Test Size':>12} | {'Features':>10} | {'Config Dim':>10} | {'Status':>10}"
    )
    print("-" * 100)

    for p_path in problem_files:
        try:
            # 1. Load raw config to get necessary params for context
            raw_p_config = load_config_with_inheritance(p_path)
            dataset_name = raw_p_config.get("dataset")

            if not dataset_name or "synthetic" in dataset_name:
                continue

            print(f"Processing {dataset_name}...")

            # Create a temporary params dict for pre-loading data to establish context
            raw_dataset_params = raw_p_config.get("dataset_params", {})
            temp_params = {
                k: v for k, v in raw_dataset_params.items() if not isinstance(v, str) or not v.startswith("expr:")
            }
            if "init_size" not in temp_params:
                temp_params["init_size"] = 0  # Default

            preloaded_data = load_dataset_from_source(dataset_name=dataset_name, random_state=0, **temp_params)

            # 2. Build the context for expression evaluation
            model_params = raw_p_config.get("model_params", {})
            n_features = preloaded_data["number_features"]
            n_train = preloaded_data["n_train"]
            param_dim = n_features + 1 if model_params.get("bias", False) else n_features
            context = {"d": param_dim, "n": n_train}

            # 3. Process the config fully with the context
            final_p_config = process_config_values(raw_p_config, context)
            final_dataset_params = final_p_config.get("dataset_params", {})

            # 4. Load the data *again* with the final, correct parameters to get the true split sizes
            final_loaded_data = load_dataset_from_source(
                dataset_name=dataset_name, random_state=0, **final_dataset_params
            )

            # 5. Perform the check and print the final, correct information
            actual_features = final_loaded_data["number_features"]
            expected_dim = actual_features + 1 if model_params.get("bias", False) else actual_features
            config_dim = final_p_config.get("dataset_params", {}).get("param_dim")

            status = "OK" if config_dim is not None and config_dim == expected_dim else "MISMATCH"

            print(
                f"{dataset_name:<20} | {final_loaded_data['n_train']:>12,} | {final_dataset_params.get('init_size', 0):>12,} | {final_loaded_data['n_test']:>12,} | {actual_features:>10} | {config_dim if config_dim is not None else 'N/A':>10} | {status:>10}"
            )

        except Exception as e:
            print(f"Could not process file {p_path}. Error: {e}")
            import traceback

            traceback.print_exc()

    print("-" * 100)


if __name__ == "__main__":
    main()
