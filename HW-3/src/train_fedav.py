import numpy as np
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from IPython import display

import os

THIS_DIR = os.path.dirname(__file__)            # .../HW-3/src
ROOT_DIR = os.path.dirname(THIS_DIR)            # .../HW-3     ✅
RESULTS_DIR = os.path.join(ROOT_DIR, "results") # .../HW-3/results
os.makedirs(RESULTS_DIR, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_federated_emnist(train_path="src/Assignment3-data/train_data.npy", test_path="src/Assignment3-data/test_data.npy"):
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)

    clients = {}
    for cid in range(len(train_data)):
        x = np.array(train_data[cid]["images"])   # convert list → ndarray
        y = np.array(train_data[cid]["labels"])
        clients[cid] = {"x": x, "y": y}

    test_data = {
        "x": np.array(test_data[0]["images"]),
        "y": np.array(test_data[0]["labels"])
    }
    return clients, test_data, train_data



def split_client_data(clients, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    train_data = {}
    val_data = {}

    for uid, data in clients.items():
        x, y = data["x"], data["y"]
        n = len(y)
        idx = np.arange(n)
        rng.shuffle(idx)
        split = int((1 - val_ratio) * n)
        train_idx = idx[:split]
        val_idx = idx[split:]

        train_data[uid] = {
            "x": x[train_idx],
            "y": y[train_idx]
        }
        val_data[uid] = {
            "x": x[val_idx],
            "y": y[val_idx]
        }

    return train_data, val_data

clients, test_data, raw_train_data = load_federated_emnist()
train_clients, val_clients = split_client_data(clients)

def compute_global_mean_std(train_clients):
    xs = []
    for cid, data in train_clients.items():
        # data["x"] has shape (Ni, 28, 28)
        x = data["x"].astype(np.float32) / 255.0   # scale once here
        xs.append(x.reshape(-1, 28*28))            # flatten per image

    all_x = np.concatenate(xs, axis=0)   # shape (N_total, 784)
    mean = all_x.mean()
    std = all_x.std()
    return float(mean), float(std)

global_mean, global_std = compute_global_mean_std(train_clients)
print("Global mean:", global_mean, "Global std:", global_std)

class SimpleFFNN(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, num_classes=62):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28) or (B, 28, 28)
        x = x.view(x.size(0), -1)       # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ClientDataset(Dataset):
    def __init__(self, x, y, mean, std):
        x = x.astype(np.float32) / 255.0
        self.x = (torch.from_numpy(x) - mean) / (std + 1e-8)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx]    # shape (28, 28)
        return img.unsqueeze(0), self.y[idx]  # (1, 28, 28)

def make_client_loaders(train_clients, val_clients, mean, std, batch_size=32):
    train_loaders = {}
    val_loaders = {}
    for uid in train_clients.keys():
        train_ds = ClientDataset(train_clients[uid]["x"], train_clients[uid]["y"], mean, std)
        val_ds   = ClientDataset(val_clients[uid]["x"],   val_clients[uid]["y"],   mean, std)
        train_loaders[uid] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loaders[uid]   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loaders, val_loaders

global_mean, global_std = compute_global_mean_std(train_clients)
train_loaders, val_loaders = make_client_loaders(train_clients, val_clients, global_mean, global_std)


def client_update(model, optimizer, train_loader, epochs, device="cpu"):
    # DO NOT deepcopy here
    # model = copy.deepcopy(model)

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    num_samples = len(train_loader.dataset)
    return model.state_dict(), num_samples

def fedavg(agg_updates):
    """
    agg_updates: list of (state_dict, num_samples)
    """
    total_samples = sum(n for _, n in agg_updates)

    # initialize with zeros using the first model’s structure
    avg_state = {}
    for key in agg_updates[0][0].keys():
        avg_state[key] = sum(
            (state[key] * (n / total_samples) for state, n in agg_updates)
        )

    return avg_state

def evaluate_global(model, loaders_dict, device="cpu"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for uid, loader in loaders_dict.items():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def live_plot(history, plot_path):
    rounds = history["round"]
    if len(rounds) == 0:
        return

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(rounds, history["train_loss"], label="Train loss")
    plt.plot(rounds, history["val_loss"], label="Val loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Rounds")

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(rounds, history["train_acc"], label="Train acc")
    plt.plot(rounds, history["val_acc"], label="Val acc")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs Rounds")

    plt.tight_layout()

    # Save (overwrite) the same file each round
    plt.tight_layout()
    plt.savefig(plot_path)   # save single updated plot
    plt.close()

def evaluate_loader(model, loader, device="cpu"):
        """Evaluate a model on a single DataLoader (used for TEST)."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
    
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
    
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
    
                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
    
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    
def train_fedavg(global_model, train_loaders, val_loaders, num_rounds=50,
                 frac_clients=0.1, local_epochs=1, lr=0.01, device="cpu",plot_path=None):

    client_ids = list(train_loaders.keys())
    m = max(1, int(frac_clients * len(client_ids)))

    history = {
        "round": [],
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    for rnd in range(num_rounds):
        # 1. sample clients
        selected = random.sample(client_ids, m)

        # 2. broadcast current global weights
        global_weights = global_model.state_dict()

        agg_updates = []
        for uid in selected:
            local_model = SimpleFFNN().to(device)
            local_model.load_state_dict(global_weights)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        
            w_new, n_k = client_update(local_model, optimizer, train_loaders[uid],
                                       epochs=local_epochs, device=device)
            agg_updates.append((w_new, n_k))

        # 3. aggregate
        new_global_weights = fedavg(agg_updates)
        global_model.load_state_dict(new_global_weights)

        # 4. evaluate global model on all train + val clients
        train_loss, train_acc = evaluate_global(global_model, train_loaders, device)
        val_loss, val_acc = evaluate_global(global_model, val_loaders, device)

        # 5. update history
        history["round"].append(rnd + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 6. live plot
        if plot_path is not None:
            live_plot(history, plot_path)

        # 7. pretty logging
        print(f"Round {rnd+1}/{num_rounds} | "
              f"Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}")

    return global_model, history
    




if __name__ == "__main__":

    configs = [
        (0.10, 10)
    ]
    
    results = {}
    
    for frac_c, E in configs:
        print(f"\n=== Running FedAvg with C={frac_c}, E={E} ===")
        global_model = SimpleFFNN().to(device)
        
        # Create directory for this config
        config_dir = f"results/C{frac_c}_E{E}"
        os.makedirs(config_dir, exist_ok=True)
        plot_path = os.path.join(config_dir, "live_plot.png")
    
        trained_model, history = train_fedavg(
            global_model,
            train_loaders,
            val_loaders,
            num_rounds=50,
            frac_clients=frac_c,
            local_epochs=E,
            lr=0.01,
            device=device,
            plot_path=plot_path,        # NEW
        )
    
        results[(frac_c, E)] = {
            "train_acc": history["train_acc"][-1],
            "val_acc": history["val_acc"][-1],
            "history": history,
        }
        
    
        
    
        # 1) Load federated train + test from your .npy files
        clients, test_data, raw_train_data = load_federated_emnist()
    
        # 2) Split each client into train/val
        train_clients, val_clients = split_client_data(clients)
    
        # 3) Compute global mean/std from TRAIN clients only
        global_mean, global_std = compute_global_mean_std(train_clients)
        print("Global mean:", global_mean, "Global std:", global_std)
    
        # 4) Build federated train/val loaders (using the same mean/std)
        train_loaders, val_loaders = make_client_loaders(
            train_clients, val_clients, mean=global_mean, std=global_std
        )
    
        # 5) Build test loader from test_data.npy (held-out test)
        test_ds = ClientDataset(test_data["x"], test_data["y"],
                            mean=global_mean, std=global_std)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
        # -------------------------------
        # 6) Train FedAvg with BEST hparams
        # -------------------------------
        best_C = 0.10   # fraction of clients
        best_E = 10     # local epochs
        best_rounds = 50
        best_lr = 0.01  # or whatever you used for Adam
    
        print(f"\n=== Training BEST FedAvg model: C={best_C}, E={best_E} ===")
        best_model = SimpleFFNN().to(device)
    
        best_model, best_history = train_fedavg(
            best_model,
            train_loaders,
            val_loaders,
            num_rounds=best_rounds,
            frac_clients=best_C,
            local_epochs=best_E,
            lr=best_lr,
            device=device,
            plot_path=os.path.join("results", "best_C0.1_E10.png"),
        )
        save_path = "results/best_fedavg_C0.1_E10.pth"
        torch.save(best_model.state_dict(), save_path)
        print("Saved trained model to:", save_path)
        
        # Optionally reload (simulating fresh process) and evaluate
        model_for_test = SimpleFFNN().to(device)
        model_for_test.load_state_dict(torch.load(save_path))
    
        # 8) Evaluate on held-out TEST data
        test_loss, test_acc = evaluate_loader(model_for_test, test_loader, device=device)
        print(f"\nFinal TEST performance (C={best_C}, E={best_E}):")
        print(f"  Test loss: {test_loss:.4f}")
        print(f"  Test acc : {test_acc:.4f}")