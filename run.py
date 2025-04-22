import torch
import wandb
import time
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Centralize hyperparameters
config = {
    "project_name": "sgd_comparison_linear_regression",
    "entity": "gsalle",
    "data": {
        "n_samples": 1000,
        "n_features": 10,
        "noise_std": 0.5,
        "train_split": 0.8,
    },
    "model": {
        "input_dim": 10,  # Should match data.n_features
        "output_dim": 1,
    },
    "training": {
        "epochs": 50,
        "batch_size": 32,  # For SGD/Mini-batch GD
        "loss_function": "MSELoss",
    },
    "optimizer1": {
        "type": "SGD",
        "lr": 0.01,
        "momentum": 0.0,
    },
    "optimizer2": {
        "type": "SGD",
        "lr": 0.001,
        "momentum": 0.9,  # Example: SGD with momentum
    },
    "simulation": {
        "num_runs": 5,  # Number of times to repeat each experiment config
        "base_seed": 42,
    },
}


import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def train_model(run_config, X_train, y_train, X_val, y_val, seed):
    """Trains a linear regression model and logs metrics to WandB."""

    # --- Reproducibility ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Initialize WandB ---
    run = wandb.init(
        project=config["project_name"],
        entity=config["entity"],
        config=run_config,  # Log specific config for this run (optimizer params, seed)
        group=f"opt_{run_config['optimizer']['type']}_lr_{run_config['optimizer']['lr']}_mom_{run_config['optimizer'].get('momentum', 0.0)}",  # Group runs by configuration
        job_type="train",
        name=f"run_{seed}",  # Name run by its seed within the group
    )

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    model = LinearRegression(config["model"]["input_dim"], config["model"]["output_dim"]).to(device)

    loss_fn = nn.MSELoss()

    # --- Select Optimizer ---
    opt_params = run_config["optimizer"]
    if opt_params["type"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_params["lr"],
            momentum=opt_params.get("momentum", 0.0),  # Use .get for optional params
        )
    # Add elif blocks here for other optimizers (e.g., Adam, LBFGS for second-order)
    # elif opt_params['type'] == 'LBFGS':
    #    optimizer = torch.optim.LBFGS(model.parameters(), lr=opt_params['lr']) # Note: LBFGS needs closure
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_params['type']}")

    # WandB: Watch the model gradients and parameters (optional)
    wandb.watch(model, log="all", log_freq=100)

    # --- Training Loop ---
    print(f"Starting training run {seed} with optimizer: {run_config['optimizer']}")
    batch_size = config["training"]["batch_size"]
    num_batches = (X_train.shape[0] + batch_size - 1) // batch_size  # Handle last batch

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        permutation = torch.randperm(X_train.size()[0])

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i : i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            # --- LBFGS requires a closure ---
            # if run_config['optimizer']['type'] == 'LBFGS':
            #     def closure():
            #         optimizer.zero_grad()
            #         outputs = model(batch_X)
            #         loss = loss_fn(outputs, batch_y)
            #         loss.backward()
            #         return loss
            #     loss = optimizer.step(closure)
            # else: # Standard SGD/Adam step
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / num_batches

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = loss_fn(val_outputs, y_val).item()

        # --- Logging ---
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                # Add other metrics: gradient norms, parameter norms, etc.
                # "grad_norm": compute_grad_norm(model), # Example
            }
        )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # --- Finish Run ---
    print(f"Finished training run {seed}.")
    run.finish()  # Crucial to finish the WandB run


if __name__ == "__main__":
    # --- Generate Base Data ---
    # Generate data once or per run depending on your experimental design
    # Here: generate once, use different seeds for training init/order
    X, y, true_w, true_b = generate_linear_data(
        config["data"]["n_samples"],
        config["data"]["n_features"],
        config["data"]["noise_std"],
        seed=config["simulation"]["base_seed"],  # Seed for data generation consistency
    )

    # --- Split Data (Best Practice) ---
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=config["data"]["train_split"],
        random_state=config["simulation"]["base_seed"],  # Use base seed for consistent splits
    )

    # --- Define Experiment Configurations ---
    optimizer_configs = [config["optimizer1"], config["optimizer2"]]
    # Add more optimizer configs here or load from a file

    # --- Run Experiments ---
    for opt_config in optimizer_configs:
        print(f"\n=== Running Experiments for Optimizer: {opt_config} ===")
        # Combine general config with specific optimizer config
        run_base_config = config.copy()
        run_base_config["optimizer"] = opt_config

        for i in range(config["simulation"]["num_runs"]):
            run_seed = config["simulation"]["base_seed"] + i + 1  # Use different seeds for each run

            # Pass only relevant parts of config to the training function
            current_run_config = {
                "seed": run_seed,
                "optimizer": opt_config,
                # Include other relevant training params if they vary per run
                "epochs": config["training"]["epochs"],
            }

            train_model(
                run_config=current_run_config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                seed=run_seed,
            )

    print("\nAll experiments finished.")
