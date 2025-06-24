import argparse
import glob
import subprocess
import os
import wandb
import pandas as pd
import time
import hashlib
from typing import List, Any, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

from datasets import load_dataset_from_source  # Needed for param_dim inference
from utils import (
    load_and_process_config,
    config_to_stable_string,
    sanitize_for_wandb,
    load_completion_log,
    expand_file_patterns,
)

# Specify the WandB entity to use for logging.
# Set to your username or a team name (e.g., "USNA").
# If set to None, the script will automatically use your default entity.
WANDB_ENTITY = "USNA"


# ============================================================================ #
# >>> Main table generation logic <<<                                          #
# ============================================================================ #


def check_experiments_completion(
    problem_files: List[str], optimizer_files: List[str], num_seeds: int
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Checks the completion log for all experiment configurations.

    Returns:
        A tuple containing:
        - A list of dictionaries for completed runs, ready to be fetched.
        - A list of string descriptions for missing runs.
    """
    completion_log_data = load_completion_log(".completed_runs.log")
    runs_to_fetch = []
    missing_run_descriptions = []

    for p_path in problem_files:
        p_name = os.path.basename(p_path).replace(".yaml", "")
        try:
            p_config = load_and_process_config(p_path, {})
            param_dim = p_config.get("dataset_params", {}).get("param_dim")
            if not param_dim:
                dataset_name = p_config.get("dataset")
                loaded_data = load_dataset_from_source(
                    dataset_name=dataset_name, random_state=0, **p_config.get("dataset_params", {})
                )
                num_features = loaded_data["number_features"]
                bias = p_config.get("model_params", {}).get("bias", False)
                param_dim = num_features + 1 if bias else num_features
            p_config["dataset_params"]["param_dim"] = param_dim
            context = {"d": param_dim, "n": int(float(p_config.get("dataset_params", {}).get("n_dataset", 1e6)))}
            project_runs_from_log = completion_log_data.get(p_name)

            for o_path in optimizer_files:
                o_name = os.path.basename(o_path).replace(".yaml", "")
                o_config = load_and_process_config(o_path, context)
                for seed in range(num_seeds):
                    current_run_config = {**p_config, **o_config, "seed": seed}
                    completion_id_stable_string = config_to_stable_string(current_run_config)
                    completion_id_hash = hashlib.md5(completion_id_stable_string.encode()).hexdigest()[:8]
                    sanitized_o_name = sanitize_for_wandb(o_name)
                    unique_run_id = f"{sanitized_o_name}_{completion_id_hash}_{seed}"
                    actual_wandb_id = project_runs_from_log.get(unique_run_id) if project_runs_from_log else None
                    run_info = {"p_name": p_name, "o_name": o_name, "seed": seed, "wandb_id": actual_wandb_id}
                    if actual_wandb_id:
                        runs_to_fetch.append(run_info)
                    else:
                        missing_run_descriptions.append(f"  - Problem: '{p_name}', Optimizer: '{o_name}', Seed: {seed}")
        except Exception as e:
            print(f"!! Failed to process config for problem {p_name} during pre-check. Error: {e}")
            continue

    return runs_to_fetch, missing_run_descriptions


def generate_accuracy_table(runs_to_fetch: List[Dict[str, Any]]) -> pd.DataFrame:
    print("\n--- Fetching results from W&B for accuracy table... ---")
    entity_to_use = WANDB_ENTITY
    if not entity_to_use:
        try:
            entity_to_use = wandb.api.default_entity
        except AttributeError:
            print("--- Could not determine default wandb entity. You may need to log in. ---")
            return pd.DataFrame()

    api = wandb.Api(timeout=20)
    results_by_config = defaultdict(lambda: defaultdict(lambda: {"train_accs": [], "test_accs": []}))
    not_found_runs = []

    for run_info in runs_to_fetch:
        p_name = run_info["p_name"]
        o_name = run_info["o_name"]
        sanitized_p_name = sanitize_for_wandb(p_name)
        run_path = f"{entity_to_use}/{sanitized_p_name}/{run_info['wandb_id']}"
        try:
            print(f"Fetching run: {run_path}")
            run = api.run(run_path)
            summary = run.summary
            summary_retries = 10
            while ("train_accuracy" not in summary or "test_accuracy" not in summary) and summary_retries > 0:
                print(f"  ...metrics not found, retrying in 10s ({summary_retries} retries left)")
                time.sleep(10)
                run.scan_history()
                summary = run.summary
                summary_retries -= 1
            if "train_accuracy" in summary and "test_accuracy" in summary:
                results_by_config[p_name][o_name]["train_accs"].append(summary["train_accuracy"])
                results_by_config[p_name][o_name]["test_accs"].append(summary["test_accuracy"])
            else:
                print(f"Warning: Could not find train/test accuracy for {run_path}. Skipping.")
        except wandb.errors.CommError:
            print(f"Could not find run {run_path} on wandb servers.")
            not_found_runs.append(run_info)
        except Exception as e:
            print(f"   An unexpected error occurred while fetching {run_path}: {e}")

    all_results = []
    for p_name, optimizers in results_by_config.items():
        for o_name, results in optimizers.items():
            train_accs, test_accs = results["train_accs"], results["test_accs"]
            if train_accs and test_accs:
                avg_train_acc = sum(train_accs) / len(train_accs)
                avg_test_acc = sum(test_accs) / len(test_accs)
                all_results.append(
                    {"Dataset": p_name, "Set": "Train", "Optimizer": o_name, "Accuracy": avg_train_acc * 100}
                )
                all_results.append(
                    {"Dataset": p_name, "Set": "Test", "Optimizer": o_name, "Accuracy": avg_test_acc * 100}
                )

    if not_found_runs:
        print("\n--- The following runs were in the log but not found on W&B ---")
        for run in not_found_runs:
            print(f"  - Problem: {run['p_name']}, Optimizer: {run['o_name']}, Seed: {run['seed']}")
        print("You may need to delete '.completed_runs.log' and re-run experiments if these runs are invalid.")

    if not all_results:
        print("No results found.")
        return pd.DataFrame()
    return pd.DataFrame(all_results)


def generate_plots(runs_to_fetch: List[Dict[str, Any]]):
    print("\n--- Fetching results from W&B for plotting... ---")
    entity_to_use = WANDB_ENTITY
    api = wandb.Api(timeout=20)

    metrics_to_plot = ["estimation_error", "inv_hess_error_fro"]
    data_for_plots = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    optimizer_times = defaultdict(lambda: defaultdict(list))
    not_found_runs = []

    for run_info in runs_to_fetch:
        p_name, o_name = run_info["p_name"], run_info["o_name"]
        sanitized_p_name = sanitize_for_wandb(p_name)
        run_path = f"{entity_to_use}/{sanitized_p_name}/{run_info['wandb_id']}"
        try:
            print(f"Fetching run for plotting: {run_path}")
            run = api.run(run_path)
            # Fetch history for line plots
            history = run.history(keys=["samples"] + metrics_to_plot, pandas=True)
            for metric in metrics_to_plot:
                if metric in history.columns:
                    metric_data = history[["samples", metric]].dropna()
                    data_for_plots[p_name][metric][o_name].append(metric_data)
            # Fetch summary for bar plot
            if "optimizer_time" in run.summary:
                optimizer_times[p_name][o_name].append(run.summary["optimizer_time"])
        except wandb.errors.CommError:
            print(f"Could not find run {run_path} on wandb servers.")
            not_found_runs.append(run_info)
        except Exception as e:
            print(f"Could not fetch data for run {run_path}: {e}")

    # --- Generate and Save Plots ---
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Line plots
    for p_name, metrics in data_for_plots.items():
        for metric, optimizers in metrics.items():
            plt.figure(figsize=(10, 6))
            for o_name, dfs in optimizers.items():
                if dfs:
                    # Average across seeds
                    merged_df = pd.concat(dfs).groupby("samples").mean().reset_index()
                    plt.plot(merged_df["samples"], merged_df[metric], label=o_name)
            plt.xlabel("Samples")
            plt.ylabel(metric)
            plt.title(f"{metric} for {p_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, f"{p_name}_{metric}.png"))
            plt.close()

    # Bar plot for optimizer times
    for p_name, optimizers in optimizer_times.items():
        avg_times = {o_name: sum(times) / len(times) for o_name, times in optimizers.items() if times}
        if avg_times:
            df = pd.DataFrame(list(avg_times.items()), columns=["Optimizer", "Time"])
            plt.figure(figsize=(10, 6))
            plt.bar(df["Optimizer"], df["Time"])
            plt.ylabel("Average Optimizer Time (s)")
            plt.title(f"Average Optimizer Time for {p_name}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{p_name}_optimizer_time.png"))
            plt.close()

    if not_found_runs:
        print("\n--- The following runs were in the log but not found on W&B ---")
        for run in not_found_runs:
            print(f"  - Problem: {run['p_name']}, Optimizer: {run['o_name']}, Seed: {run['seed']}")
        print("You may need to delete '.completed_runs.log' and re-run experiments if these runs are invalid.")

    print(f"Plots saved to '{plot_dir}/' directory.")


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pivot_df = df.pivot_table(index=["Dataset", "Set"], columns="Optimizer", values="Accuracy")
    pivot_df = pivot_df.reset_index().set_index("Dataset")
    return pivot_df.round(2)


def main():
    parser = argparse.ArgumentParser(description="Generate a results table or plots from completed experiments.")
    parser.add_argument(
        "-a",
        "--action",
        choices=["table", "plot"],
        required=True,
        help="Action to perform: generate a 'table' or a 'plot'.",
    )
    parser.add_argument(
        "-p",
        "--problems",
        nargs="+",
        required=True,
        help="Path(s) to YAML problem configuration files. Wildcards are supported.",
    )
    parser.add_argument(
        "-o",
        "--optimizers",
        nargs="+",
        required=True,
        help="Path(s) to YAML optimizer configuration files. Wildcards are supported.",
    )
    parser.add_argument(
        "-N", "--num-seeds", type=int, default=1, help="Number of random seeds to run for each experiment."
    )
    args = parser.parse_args()

    problem_files = expand_file_patterns(args.problems)
    optimizer_files = expand_file_patterns(args.optimizers)

    if not problem_files or not optimizer_files:
        print("No problem or optimizer configuration files found.")
        return

    print("\n--- Checking for completed experiments... ---")
    runs_to_fetch, missing_runs = check_experiments_completion(problem_files, optimizer_files, args.num_seeds)

    if missing_runs:
        print("\n--- Missing Experiments ---")
        print("Could not proceed because the following experiment runs are missing:")
        for desc in missing_runs:
            print(desc)
        print("\nPlease run the required experiments using 'main.py'.")
        problems_str = " ".join(f'"{p}"' for p in args.problems)
        optimizers_str = " ".join(f'"{o}"' for o in args.optimizers)
        print(
            f"Command to run all experiments: python main.py -p {problems_str} -o {optimizers_str} --num-seeds {args.num_seeds}"
        )
        return

    print("\n--- All experiments are completed. ---")

    if args.action == "plot":
        generate_plots(runs_to_fetch)
    elif args.action == "table":
        results_df = generate_accuracy_table(runs_to_fetch)
        if not results_df.empty:
            final_table = format_table(results_df)
            print("\n--- Results Table ---")
            print(final_table.to_string())


if __name__ == "__main__":
    main()
