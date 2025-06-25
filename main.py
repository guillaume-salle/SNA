import os
import wandb
import traceback
import hashlib
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from typing import List, Dict, Any

from utils import (
    RunCompletionManager,
    load_and_process_config,
    config_to_stable_string,
    sanitize_for_wandb,
    expand_file_patterns,
)
from run import run_experiment
from datasets import load_dataset_from_source


# Specify the WandB entity to use for logging.
# Set to your username or a team name (e.g., "USNA").
# If set to None, the script will automatically use your default entity.
WANDB_ENTITY = "USNA"


# ============================================================================ #
# >>> Visualization Functions <<<
# ============================================================================ #


def parse_local_run(local_dir: str, metrics_to_plot: List[str]) -> Dict | None:
    """Parses summary and history files from a local wandb run directory."""
    if not local_dir or not os.path.isdir(local_dir):
        return None

    summary_file = os.path.join(local_dir, "files", "wandb-summary.json")
    history_file = os.path.join(local_dir, "files", "wandb-history.jsonl")

    run_data: Dict[str, Any] = {"history": {}}

    # Parse summary file for final metrics
    try:
        with open(summary_file, "r") as f:
            summary_data = json.load(f)
            for key in ["final_train_accuracy", "final_test_accuracy", "optimizer_time"]:
                if key in summary_data:
                    run_data[key] = summary_data[key]
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        # It's possible the summary hasn't been written yet, which is okay.
        pass

    # Parse history file for line plots
    try:
        metric_history = {metric: [] for metric in metrics_to_plot}
        with open(history_file, "r") as f:
            for line in f:
                history_item = json.loads(line)
                # Only log items that have a "samples" key
                if "samples" in history_item:
                    for metric in metrics_to_plot:
                        if metric in history_item:
                            metric_history[metric].append(
                                {"samples": history_item["samples"], metric: history_item[metric]}
                            )
        for metric, data in metric_history.items():
            if data:
                run_data["history"][metric] = data
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        pass  # History might not exist or be complete, which is fine.

    return run_data


def generate_accuracy_table(runs: List[Dict[str, Any]]):
    print("\n--- Generating accuracy table from local run data... ---")
    results_by_config = defaultdict(
        lambda: defaultdict(lambda: {"train_accs": [], "test_accs": [], "optimizer_times": []})
    )

    for run_info in runs:
        p_name, o_name = run_info["p_name"], run_info["o_name"]
        local_data = parse_local_run(run_info["local_dir"], [])
        if not local_data:
            print(f"Warning: Could not parse local data for run {run_info['wandb_id']}. Skipping.")
            continue

        if (
            "final_train_accuracy" in local_data
            and "final_test_accuracy" in local_data
            and "optimizer_time" in local_data
        ):
            results_by_config[p_name][o_name]["train_accs"].append(local_data["final_train_accuracy"])
            results_by_config[p_name][o_name]["test_accs"].append(local_data["final_test_accuracy"])
            results_by_config[p_name][o_name]["optimizer_times"].append(local_data["optimizer_time"])
        else:
            print(f"Warning: Missing required metrics in local files for run {run_info['wandb_id']}. Skipping.")

    all_results = []
    for p_name, optimizers in results_by_config.items():
        for o_name, results in optimizers.items():
            train_accs, test_accs, optimizer_times = (
                results["train_accs"],
                results["test_accs"],
                results["optimizer_times"],
            )
            if train_accs and test_accs and optimizer_times:
                avg_train_acc = sum(train_accs) / len(train_accs)
                avg_test_acc = sum(test_accs) / len(test_accs)
                avg_time = sum(optimizer_times) / len(optimizer_times)
                all_results.append(
                    {
                        "Dataset": p_name,
                        "Set": "Train",
                        "Optimizer": o_name,
                        "Accuracy": avg_train_acc * 100,
                        "Time (s)": avg_time,
                    }
                )
                all_results.append(
                    {
                        "Dataset": p_name,
                        "Set": "Test",
                        "Optimizer": o_name,
                        "Accuracy": avg_test_acc * 100,
                        "Time (s)": avg_time,
                    }
                )

    if not all_results:
        print("No results to display.")
        return

    final_table = format_table(pd.DataFrame(all_results))
    print("\n--- Results Table ---")
    print(final_table.to_string())


def generate_plots(runs: List[Dict[str, Any]], metrics_to_plot: List[str]):
    print("\n--- Generating plots from local run data... ---")
    data_for_plots = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    optimizer_times = defaultdict(lambda: defaultdict(list))

    for run_info in runs:
        p_name, o_name = run_info["p_name"], run_info["o_name"]
        local_data = parse_local_run(run_info["local_dir"], metrics_to_plot)
        if not local_data:
            print(f"Warning: Could not parse local data for run {run_info['wandb_id']}. Skipping.")
            continue

        local_history = local_data.get("history", {})
        # Line plots
        for metric in metrics_to_plot:
            if metric in local_history and local_history[metric]:
                metric_df = pd.DataFrame.from_records(local_history[metric])
                data_for_plots[p_name][metric][o_name].append(metric_df)
        # Bar plot for optimizer times
        if "optimizer_time" in local_data:
            optimizer_times[p_name][o_name].append(local_data["optimizer_time"])

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

    print(f"Plots saved to '{plot_dir}/' directory.")


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pivot_df = df.pivot_table(index=["Dataset", "Set"], columns="Optimizer", values=["Accuracy", "Time (s)"])
    pivot_df = pivot_df.reset_index().set_index("Dataset")
    return pivot_df.round(2)


def run_visualizations(runs_to_fetch: List[Dict[str, Any]], problem_files: List[str]):
    """Determines which visualization to run based on the problem type."""
    metrics_to_plot = ["estimation_error", "inv_hess_error_fro"]

    is_synthetic_list = ["synthetic" in p for p in problem_files]
    if all(is_synthetic_list):
        is_synthetic = True
    elif not any(is_synthetic_list):
        is_synthetic = False
    else:
        raise ValueError("All problems must be either synthetic or all not synthetic. Mixed types detected.")

    if is_synthetic:
        generate_plots(runs_to_fetch, metrics_to_plot)
    else:
        generate_accuracy_table(runs_to_fetch)


def main():
    """
    Main function to run experiments, check completion, and visualize results.
    """
    parser = argparse.ArgumentParser(description="Run optimization experiments.")
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
        "-N",
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds to run for each experiment.",
    )
    parser.add_argument(
        "--check-runs",
        action="store_true",
        help="Only check for completed runs and report missing ones without running them.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visuals (table or plot) if all experiments are complete. Does not run experiments.",
    )
    args = parser.parse_args()

    problem_files = expand_file_patterns(args.problems)
    optimizer_files = expand_file_patterns(args.optimizers)

    if not problem_files:
        print("No problem configuration files found from the provided patterns.")
        return
    if not optimizer_files:
        print("No optimizer configuration files found from the provided patterns.")
        return

    # --- Set up completion manager ---
    completion_manager = RunCompletionManager()
    completion_log_data = completion_manager.get_log_data()
    runs_to_fetch = []
    missing_runs = []

    for p_path in problem_files:
        p_name = os.path.basename(p_path).replace(".yaml", "")
        p_config = load_and_process_config(p_path, {})

        # --- Create Context for Optimizer Configs ---
        dataset_params = p_config.get("dataset_params")
        if not dataset_params:
            raise ValueError(f"Problem config '{p_name}' is missing 'dataset_params' field.")

        param_dim: int | None = None
        # Infer param_dim if not specified in the config
        if "param_dim" in dataset_params:
            param_dim = dataset_params["param_dim"]
        elif "true_theta" in dataset_params:
            param_dim = len(dataset_params["true_theta"])
        else:
            if "synthetic" in p_name:
                raise ValueError(
                    f"Problem '{p_name}' is synthetic data, so param_dim or true_theta must be specified in the config."
                )
            print(f"   Inferring param_dim for problem '{p_name}' by loading the dataset...")
            dataset_name = p_config.get("dataset")
            if not dataset_name:
                raise ValueError(f"Problem config '{p_name}' is missing 'dataset' field.")

            loaded_data = load_dataset_from_source(dataset_name=dataset_name, random_state=0, **dataset_params)
            num_features = loaded_data["number_features"]
            bias = p_config.get("model_params", {}).get("bias", False)
            param_dim = num_features + 1 if bias else num_features
            print(f"   Inferred param_dim = {param_dim}")

        p_config["dataset_params"]["param_dim"] = param_dim
        context = {"d": param_dim, "n": dataset_params.get("n_dataset")}

        for o_path in optimizer_files:
            o_name = os.path.basename(o_path).replace(".yaml", "")
            o_config = load_and_process_config(o_path, context)

            for i in range(args.num_seeds):
                seed = i

                # --- Create unique run ID and check for completion ---
                current_run_config = {**p_config, **o_config, "seed": seed}
                completion_id_stable_string = config_to_stable_string(current_run_config)
                completion_id_hash = hashlib.md5(completion_id_stable_string.encode()).hexdigest()[:8]
                sanitized_o_name = sanitize_for_wandb(o_name)
                unique_run_id = f"{sanitized_o_name}_{completion_id_hash}_{seed}"

                completed_run_data = completion_log_data.get(p_name, {}).get(unique_run_id)

                if completed_run_data and completed_run_data.get("local_dir"):
                    # Run is complete and has a valid local directory
                    run_info = {
                        "p_name": p_name,
                        "o_name": o_name,
                        "seed": seed,
                        "wandb_id": completed_run_data["wandb_id"],
                        "local_dir": completed_run_data["local_dir"],
                    }
                    runs_to_fetch.append(run_info)
                    continue

                # --- At this point, the run is MISSING or has an invalid log entry ---

                if args.check_runs or args.visualize:
                    # In a "don't run" mode, just record it and move on.
                    missing_runs.append(f"Problem='{p_name}', Optimizer='{o_name}', Seed={seed}")
                    continue

                # --- EXECUTION MODE for a missing run ---
                print(f"--- Running missing experiment: Problem='{p_name}', Optimizer='{o_name}', Seed={seed} ---")
                wandb_run = None
                success = False
                try:
                    entity_to_use = WANDB_ENTITY
                    if not entity_to_use:
                        try:
                            entity_to_use = wandb.api.default_entity
                            print(f"--- Using default wandb entity: '{entity_to_use}' ---")
                        except AttributeError:
                            print(
                                "--- Could not determine default wandb entity. You may need to log in. Letting wandb decide. ---"
                            )
                            entity_to_use = None
                    sanitized_p_name = sanitize_for_wandb(p_name)
                    group_name = sanitized_o_name
                    wandb_run_name = f"{sanitized_o_name}"

                    print(f"--- Starting run: {wandb_run_name} (Project: {sanitized_p_name}, Group: {group_name}) ---")

                    wandb_run = wandb.init(
                        entity=entity_to_use,
                        project=sanitized_p_name,
                        config=current_run_config,
                        group=group_name,
                        name=wandb_run_name,
                        mode="online",
                    )
                    actual_wandb_id = wandb_run.id
                    local_dir = wandb_run.dir
                    run_experiment(p_config, o_config, seed=i, project_name=p_name)

                    success = True
                    completion_manager.log_run_completion(unique_run_id, actual_wandb_id, local_dir, p_name)
                    print(f"--- Finished and logged run (ID: {unique_run_id} / wandb: {actual_wandb_id}) ---")

                    # Add the newly completed run to our list for potential visualization
                    if args.visualize:
                        run_info = {
                            "p_name": p_name,
                            "o_name": o_name,
                            "seed": seed,
                            "wandb_id": actual_wandb_id,
                            "local_dir": local_dir,
                        }
                        runs_to_fetch.append(run_info)
                except Exception as e:
                    print(f"!!! ERROR during execution for {o_name}, seed {seed}: {e} !!!")
                    traceback.print_exc()
                    raise
                finally:
                    if wandb_run is not None:
                        exit_code = 0 if success else 1
                        wandb.finish(exit_code=exit_code)
                        print(f"--- WandB run finished for {o_name} (Exit code: {exit_code}) ---")

    # --- Post-Loop Actions ---
    if missing_runs:
        # This block is only reached if --check-runs or --visualize was used and runs were missing.
        print("\n--- Missing Experiments ---")
        print("The following experiment runs were not found in the completion log:")
        for desc in missing_runs:
            print(f"  - {desc}")
        if args.visualize:
            print("\nCannot generate visuals because some experiments are missing. Please run them first.")
        return

    # If we get here, all specified experiments are now complete.
    if args.visualize:
        print("\n--- All experiments are completed. Proceeding with visualization... ---")
        run_visualizations(runs_to_fetch, problem_files)
    elif args.check_runs:
        print("\n--- All experiments are completed. ---")
    else:
        # This is the default case (no flags) where all runs were either already complete or were just run.
        print("\n--- All necessary experiments are now complete. ---")


if __name__ == "__main__":
    main()
