import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Reuse everything from your Part 1 code
from train_fedav import (
    load_federated_emnist,
    split_client_data,
    compute_global_mean_std,
    make_client_loaders,
    SimpleFFNN,
    ClientDataset,
    train_fedavg,
    evaluate_loader,
    RESULTS_DIR,
    device,
)

# ---------------------------------------------------
# 1) Laplace noise helper: per-client data perturbation
# ---------------------------------------------------

def add_laplace_noise_to_clients(clients, b, seed=0):
    """
    Given a dict {uid: {"x": images, "y": labels}} where x is in [0,255],
    return a NEW dict where each image is perturbed as:

        X' = X + eps,  eps ~ Lap(0, b_norm) in normalized [0,1] space.

    We implement this by sampling noise in normalized units and
    scaling up by 255, so the hyperparameter b is interpretable in [0,1].

    b : noise scale in *normalized* units (e.g., 0.0, 0.01, 0.05, 0.1).
    """
    rng = np.random.default_rng(seed)
    noisy_clients = {}

    # noise scale in pixel units
    scale_pix = b * 255.0

    for uid, data in clients.items():
        x = data["x"].astype(np.float32)   # shape (Ni, 28, 28), values ~0..255
        y = data["y"]

        eps = rng.laplace(loc=0.0, scale=scale_pix, size=x.shape).astype(np.float32)
        x_noisy = x + eps

        noisy_clients[uid] = {
            "x": x_noisy,
            "y": y,
        }

    return noisy_clients


# ---------------------------------------------------
# 2) Main DP experiment
# ---------------------------------------------------

if __name__ == "__main__":
    # A) Load raw federated data (same as Part 1)
    clients, test_data, raw_train_data = load_federated_emnist()

    # B) Split each client into train/val (ON CLEAN DATA)
    train_clients, val_clients = split_client_data(clients)

    # C) Compute global mean/std from CLEAN train data
    global_mean, global_std = compute_global_mean_std(train_clients)
    print("Global mean:", global_mean, "Global std:", global_std)

    # D) Build clean test loader (NO NOISE on test!)
    test_ds = ClientDataset(
        test_data["x"], test_data["y"],
        mean=global_mean, std=global_std
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # FedAvg hyperparams required by the assignment:
    C = 0.10          # 10% clients per round
    E = 10            # local epochs (you can use what you used in Part 1)
    num_rounds = 50
    lr = 0.01

    # Noise scales to try (you can adjust these)
    noise_scales = [0.0, 0.001, 0.002, 0.005, 0.01,0.015]

    final_train_accs = []
    final_test_accs = []

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for b in noise_scales:
        print(f"\n=== Training FedAvg with Laplace noise scale b = {b} ===")

        # E) Perturb TRAIN data ONLY (local training data)
        #    Val remains clean so we measure performance on non-noisy data.
        train_clients_noisy = add_laplace_noise_to_clients(
            train_clients, b=b, seed=0
        )

        # F) Build loaders: train uses noisy data; val uses clean data
        train_loaders, val_loaders = make_client_loaders(
            train_clients_noisy,  # noisy x
            val_clients,          # clean x
            mean=global_mean,
            std=global_std,
        )

        # G) Fresh global model for this noise scale
        model = SimpleFFNN().to(device)

        plot_path = os.path.join(
            RESULTS_DIR,
            f"dp_curve_b{b:.3f}.png"
        )

        # H) Run standard (sequential) FedAvg from Part 1
        model, history = train_fedavg(
            model,
            train_loaders,
            val_loaders,
            num_rounds=num_rounds,
            frac_clients=C,
            local_epochs=E,
            lr=lr,
            device=device,
            plot_path=plot_path,
        )

        # Final training accuracy from history
        final_train_acc = history["train_acc"][-1]
        final_train_accs.append(final_train_acc)

        # Evaluate on held-out CLEAN test data
        test_loss, test_acc = evaluate_loader(model, test_loader, device=device)
        final_test_accs.append(test_acc)

        print(f"b={b}: final train_acc={final_train_acc:.4f}, test_acc={test_acc:.4f}")

    # ---------------------------------------------------
    # 3) Plot accuracy vs. noise scale b (Part 2, Q2)
    # ---------------------------------------------------
    plt.figure()
    plt.plot(noise_scales, final_train_accs, marker="o", label="Train accuracy")
    plt.plot(noise_scales, final_test_accs, marker="s", label="Test accuracy")
    plt.xlabel("Laplace noise scale b")
    plt.ylabel("Accuracy")
    plt.title("Effect of Laplace noise on FedAvg performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    summary_plot_path = os.path.join(RESULTS_DIR, "dp_accuracy_vs_b.png")
    plt.savefig(summary_plot_path)
    plt.close()

    print("\nSaved summary DP plot to:", summary_plot_path)