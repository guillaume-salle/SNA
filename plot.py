import wandb
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict


def fetch_and_plot(project_name, entity, metrics_to_plot):
    """
    Fetches all runs from a project and plots specified metrics,
    grouping by optimizer.
    """
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project_name}")
        print(f"Found {len(runs)} runs in project '{project_name}'")

        # Group runs by their group name (which corresponds to the optimizer)
        grouped_runs = defaultdict(list)
        for run in runs:
            # The user's setup uses run.group to identify the optimizer
            if run.group:
                grouped_runs[run.group].append(run)

        # --- Process and Plot each metric ---
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 7))
            all_optimizers_had_data = False

            for optimizer_name, run_list in grouped_runs.items():
                print(f"  Processing optimizer: {optimizer_name} ({len(run_list)} seed(s))")

                all_histories = []
                for run in run_list:
                    # Download history for the specific metric + samples for the x-axis
                    history = run.history(keys=["samples", metric], pandas=True)
                    if not history.empty and metric in history.columns and not history[metric].dropna().empty:
                        # Set 'samples' as the index for easy alignment later
                        all_histories.append(history.set_index("samples")[[metric]].dropna())

                if not all_histories:
                    print(f"    -> No data found for metric '{metric}' in optimizer '{optimizer_name}'")
                    continue

                all_optimizers_had_data = True
                # Combine histories from all seeds and average them
                combined_history = pd.concat(all_histories)
                mean_history = combined_history.groupby(combined_history.index).mean()

                # Plot the averaged history
                plt.plot(mean_history.index, mean_history[metric], label=optimizer_name)

            # --- Finalize and Save the Plot ---
            if not all_optimizers_had_data:
                print(f"\nNo data found for any optimizer for metric '{metric}'. Skipping plot.")
                plt.close()
                continue

            plt.title(f"'{metric}' for Project '{project_name}'")
            plt.xlabel("Samples")
            plt.ylabel(metric)
            if "error" in metric.lower():
                plt.yscale("log")  # Use log scale for error plots
            plt.legend()
            plt.grid(True, which="both", linestyle="--")

            plot_filename = f"{project_name}_{metric}.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"\nPlot saved to '{plot_filename}'")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure project and entity names are correct and you are logged in.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results from W&B.")
    parser.add_argument("project", type=str, help="The name of the W&B project (e.g., 'linear_random_d-1000_N-1e5').")
    parser.add_argument("--entity", type=str, default="USNA", help="The W&B entity (username or team).")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["estimation_error", "inv_hess_error_fro"],
        help="A list of metrics to plot.",
    )
    args = parser.parse_args()

    fetch_and_plot(args.project, args.entity, args.metrics)
