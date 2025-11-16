# src/train_fedavg_ray.py

import os
import numpy as np
import ray
import torch
from torch.utils.data import DataLoader

# Import all the goodies from your existing file
from train_fedav import (
    load_federated_emnist,
    split_client_data,
    compute_global_mean_std,
    make_client_loaders,
    SimpleFFNN,
    ClientDataset,
    evaluate_global,
    evaluate_loader,   # for test set
    fedavg,            # FedAvg aggregator
    live_plot,
    RESULTS_DIR,       # reuse the same results dir
    device             # same device logic (cuda vs cpu)
)

# -----------------------
# 1. Ray Actor definition
# -----------------------

@ray.remote
class ClientTrainer:
    def __init__(self, client_id, model_cls, model_kwargs,
                 train_loader, lr, local_epochs, device="cpu"):
        self.client_id = client_id
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs or {}
        self.train_loader = train_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = torch.device(device)

    def train(self, global_state_dict):
        # Create a fresh model for this client
        model = self.model_cls(**self.model_kwargs).to(self.device)
        model.load_state_dict(global_state_dict)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        num_samples = 0

        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                num_samples += labels.size(0)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        # Return updated weights + how many samples this client used
        return model.state_dict(), num_samples


# -----------------------
# 2. Ray-based FedAvg loop
# -----------------------

def train_fedavg_ray(
    global_model,
    train_loaders,
    val_loaders,
    num_rounds=50,
    frac_clients=0.1,
    local_epochs=1,
    lr=0.01,
    clients_per_round=4,
    device="cpu",
    plot_path=None,
):
    client_ids = list(train_loaders.keys())
    num_clients = len(client_ids)

    history = {
        "round": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for rnd in range(1, num_rounds + 1):
        # 1) sample clients
        m = max(1, int(frac_clients * num_clients))
        m = min(m, clients_per_round)  # cap at 4
        selected = np.random.choice(client_ids, size=m, replace=False)

        # 2) broadcast current global weights
        global_state = global_model.state_dict()

        # 3) spawn Ray Actors for the selected clients
        #    Each actor: at most 1 CPU and 1 GPU
        resources = ray.available_resources()
        num_gpus = int(resources.get("GPU", 0))

        use_gpu_for_clients = num_gpus >= m  # simple policy: GPU per client if enough

        actors = []
        for uid in selected:
            actor_device = "cuda" if use_gpu_for_clients else "cpu"
            actor = ClientTrainer.options(
                num_cpus=1,
                num_gpus=1 if use_gpu_for_clients else 0,
            ).remote(
                int(uid),
                global_model.__class__,  # SimpleFFNN
                {},                      # model_kwargs
                train_loaders[uid],
                lr,
                local_epochs,
                actor_device
            )
            actors.append(actor)

        # 4) launch parallel training
        futures = [actor.train.remote(global_state) for actor in actors]
        results = ray.get(futures)  # [(state_dict_k, n_k), ...]

        # 5) aggregate via FedAvg (reusing your fedavg function)
        new_global_state = fedavg(results)
        global_model.load_state_dict(new_global_state)

        # 6) evaluate global model
        train_loss, train_acc = evaluate_global(global_model, train_loaders, device)
        val_loss, val_acc = evaluate_global(global_model, val_loaders, device)

        history["round"].append(rnd)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 7) live plot (optional)
        if plot_path is not None:
            live_plot(history, plot_path)

        # 8) logging
        print(
            f"[Ray] Round {rnd}/{num_rounds} | "
            f"Train loss: {train_loss:.3f}, acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.3f}, acc: {val_acc:.3f}"
        )

    return global_model, history


# -----------------------
# 3. Main: data, Ray init, test eval
# -----------------------

if __name__ == "__main__":
    # 1) Init Ray
    ray.init(ignore_reinit_error=True)
    print("Ray initialized with resources:", ray.available_resources())

    # 2) Load federated data
    clients, test_data, raw_train_data = load_federated_emnist()

    # 3) Split into train/val per client
    train_clients, val_clients = split_client_data(clients)

    # 4) global mean/std from train clients
    global_mean, global_std = compute_global_mean_std(train_clients)
    print("Global mean:", global_mean, "Global std:", global_std)

    # 5) build train/val loaders
    train_loaders, val_loaders = make_client_loaders(
        train_clients, val_clients, mean=global_mean, std=global_std
    )

    # 6) build test loader
    test_ds = ClientDataset(
        test_data["x"], test_data["y"],
        mean=global_mean, std=global_std
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # 7) Ray FedAvg config
    best_C = 0.10      # fraction of clients
    best_E = 10        # local epochs
    best_rounds = 50
    best_lr = 0.01

    print(f"\n=== Training Ray FedAvg model: C={best_C}, E={best_E} ===")
    best_model = SimpleFFNN().to(device)

    # make results path for Ray run
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ray_plot_path = os.path.join(RESULTS_DIR, "ray_C0.1_E10.png")

    best_model, best_history = train_fedavg_ray(
        best_model,
        train_loaders,
        val_loaders,
        num_rounds=best_rounds,
        frac_clients=best_C,
        local_epochs=best_E,
        lr=best_lr,
        clients_per_round=4,
        device=device,
        plot_path=ray_plot_path,
    )

    # 8) save model
    save_path = os.path.join(RESULTS_DIR, "ray_fedavg_C0.1_E10.pth")
    torch.save(best_model.state_dict(), save_path)
    print("Saved Ray-trained model to:", save_path)

    # 9) evaluate on held-out TEST data
    model_for_test = SimpleFFNN().to(device)
    model_for_test.load_state_dict(torch.load(save_path))
    test_loss, test_acc = evaluate_loader(model_for_test, test_loader, device=device)
    print(f"\n[Ray] Final TEST performance (C={best_C}, E={best_E}):")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test acc : {test_acc:.4f}")

    # 10) shutdown Ray
    ray.shutdown()