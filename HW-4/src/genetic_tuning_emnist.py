#!/usr/bin/env python3
"""
Genetic hyperparameter tuning for Federated EMNIST (digits only).

Search space:
  - Batch size B ∈ {16, 32, 64, 128, 256, 512, 1024}
  - Activation function ∈ {ReLU, Sigmoid, Tanh}

Fitness:
  - Macro-averaged F1 score on validation set.

Outputs:
  - results/ga_fitness.png   : avg & best fitness vs generation
  - results/train_f1_best.png: training F1 vs epoch for best hyperparams
  - Prints best hyperparams and test F1

Assumed directory structure (from HW-4 root):
  HW-4/
    Assignment3-data/
      train_data.npy
      test_data.npy
    src/
      genetic_tuning_emnist.py   <-- this file
    results/
      (plots will be saved here)
"""

import os
from pathlib import Path
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# ============================
#   PATHS / CONFIG
# ============================

THIS_DIR = Path(__file__).resolve().parent         # .../HW-4/src
ROOT_DIR = THIS_DIR.parent                         # .../HW-4
DATA_DIR = ROOT_DIR / "Assignment3-data"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_FILE = DATA_DIR / "train_data.npy"
TEST_DATA_FILE = DATA_DIR / "test_data.npy"

TRAIN_X_FILE = DATA_DIR / "train_X.npy"
TRAIN_Y_FILE = DATA_DIR / "train_y.npy"
TEST_X_FILE = DATA_DIR / "test_X.npy"
TEST_Y_FILE = DATA_DIR / "test_y.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU in use:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected.")

# GA search space
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]
ACTIVATIONS = ["relu", "sigmoid", "tanh"]

# GA hyperparameters (you can tweak these)
POP_SIZE = 12
NUM_GENERATIONS = 10
MAX_AGE = 3               # generations; younger individuals preferred
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2

# Training hyperparameters for fitness evaluation
FITNESS_EPOCHS = 15        # keep small to make GA affordable
FINAL_TRAIN_EPOCHS = 25   # train longer for final model
LEARNING_RATE = 0.01
MOMENTUM = 0.9

HIDDEN_DIM = 128
NUM_CLASSES = 10
INPUT_DIM = 28 * 28       # EMNIST images are 28x28


# ============================
#   UTILITIES
# ============================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_construct_xy_files():
    """
    Construct train_X.npy, train_y.npy, test_X.npy, test_y.npy
    from train_data.npy, test_data.npy if they don't exist.

    Assumes:
      - train_data.npy and test_data.npy each contain a 1D np.ndarray
        of dicts, where each dict has keys "images" and "labels":
            d["images"] -> (n_i, 28, 28)
            d["labels"] -> (n_i,)
    """
    if TRAIN_X_FILE.exists() and TRAIN_Y_FILE.exists() and \
       TEST_X_FILE.exists() and TEST_Y_FILE.exists():
        print("Found existing *_X.npy and *_y.npy files, using them.")
        return

    print("Constructing *_X.npy and *_y.npy from train_data.npy / test_data.npy ...")

    train_obj = np.load(TRAIN_DATA_FILE, allow_pickle=True)
    test_obj  = np.load(TEST_DATA_FILE,  allow_pickle=True)

    # ---- sanity prints ----
    print("train_data type:", type(train_obj), "shape:", getattr(train_obj, "shape", None))
    print("test_data  type:", type(test_obj),  "shape:", getattr(test_obj, "shape", None))

    # Expect 1D object arrays of dicts
    assert isinstance(train_obj, np.ndarray) and train_obj.ndim == 1, \
        f"Unexpected train_obj structure: type={type(train_obj)}, shape={getattr(train_obj, 'shape', None)}"
    assert isinstance(test_obj, np.ndarray) and test_obj.ndim == 1, \
        f"Unexpected test_obj structure: type={type(test_obj)}, shape={getattr(test_obj, 'shape', None)}"

    def collect_xy(obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for i, item in enumerate(obj):
            # item can be a dict or 0-D ndarray containing a dict
            if isinstance(item, dict):
                d = item
            elif isinstance(item, np.ndarray) and item.shape == () and isinstance(item.item(), dict):
                d = item.item()
            else:
                raise ValueError(
                    f"Unexpected element at index {i}: type={type(item)}, shape={getattr(item, 'shape', None)}"
                )

            if "images" not in d or "labels" not in d:
                raise ValueError(f"Element {i} missing 'images'/'labels' keys: keys={list(d.keys())}")

            x_i = np.array(d["images"])
            y_i = np.array(d["labels"])

            if x_i.shape[0] != y_i.shape[0]:
                raise ValueError(
                    f"Element {i} images/labels length mismatch: "
                    f"images.shape[0]={x_i.shape[0]}, labels.shape[0]={y_i.shape[0]}"
                )

            xs.append(x_i)
            ys.append(y_i)

        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return X, y

    train_X, train_y = collect_xy(train_obj)
    test_X,  test_y  = collect_xy(test_obj)

    print("Raw train_X shape:", train_X.shape)
    print("Raw train_y shape:", train_y.shape)
    print("Raw test_X  shape:", test_X.shape)
    print("Raw test_y  shape:", test_y.shape)

    # If labels already 0–9, no need to filter, but we can enforce digits just in case.
    digit_mask_train = (train_y >= 0) & (train_y <= 9)
    digit_mask_test  = (test_y  >= 0) & (test_y  <= 9)

    train_X = train_X[digit_mask_train]
    train_y = train_y[digit_mask_train]
    test_X  = test_X[digit_mask_test]
    test_y  = test_y[digit_mask_test]

    print("After digit filter:")
    print("  train_X shape:", train_X.shape)
    print("  train_y shape:", train_y.shape)
    print("  test_X  shape:", test_X.shape)
    print("  test_y  shape:", test_y.shape)
    print("  unique train labels:", np.unique(train_y))
    print("  unique test  labels:", np.unique(test_y))

    # Flatten to (N, 784)
    def reshape_flat(x: np.ndarray) -> np.ndarray:
        if x.ndim == 4 and x.shape[1] == 1:  # (N,1,28,28)
            x = x[:, 0, :, :]
        if x.ndim == 3:                      # (N,28,28)
            x = x.reshape(x.shape[0], -1)
        return x.astype(np.float32)

    train_X = reshape_flat(train_X)
    test_X  = reshape_flat(test_X)

    # Ensure dtype for labels
    train_y = train_y.astype(np.int64)
    test_y  = test_y.astype(np.int64)

    # Optional: save a few debug images to visually inspect (open later)
    try:
        import matplotlib.pyplot as plt
        for i in range(5):
            img = train_X[i].reshape(28, 28)
            plt.imshow(img, cmap="gray")
            plt.title(f"label {train_y[i]}")
            plt.axis("off")
            plt.savefig(RESULTS_DIR / f"debug_sample_{i}.png", bbox_inches="tight")
            plt.close()
        print("Saved a few debug images to results/debug_sample_*.png")
    except Exception as e:
        print("Could not save debug images:", e)

    # Save final .npy files
    np.save(TRAIN_X_FILE, train_X)
    np.save(TRAIN_Y_FILE, train_y)
    np.save(TEST_X_FILE, test_X)
    np.save(TEST_Y_FILE, test_y)

    print("Saved:")
    print(f"  {TRAIN_X_FILE}")
    print(f"  {TRAIN_Y_FILE}")
    print(f"  {TEST_X_FILE}")
    print(f"  {TEST_Y_FILE}")

def load_data_and_split() -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Load train_X/y and test_X/y, split train into 80% train / 20% val.
    Return PyTorch TensorDatasets for (train, val, test).
    """
    maybe_construct_xy_files()

    train_X = np.load(TRAIN_X_FILE).astype(np.float32)
    train_y = np.load(TRAIN_Y_FILE)
    test_X  = np.load(TEST_X_FILE).astype(np.float32)
    test_y  = np.load(TEST_Y_FILE)
    
    # Normalize to [0,1] **only if** data look like 0–255
    max_val = max(train_X.max(), test_X.max())
    print("Max pixel value in loaded data:", max_val)
    
    if max_val > 1.5:  # assume 0–255 images
        print("Scaling images by 1/255.0 (0–255 -> 0–1).")
        train_X = train_X / 255.0
        test_X  = test_X / 255.0
    else:
        print("Images already in [0,1]; skipping additional scaling.")

    N = train_X.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    split = int(0.8 * N)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = train_X[train_idx], train_y[train_idx]
    X_val, y_val = train_X[val_idx], train_y[val_idx]

    X_test, y_test = test_X, test_y

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    return train_ds, val_ds, test_ds


# ============================
#   MODEL
# ============================

class LeNet5(nn.Module):
    def __init__(self, activation: str, n_classes=10):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError("invalid activation")

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            act,
            nn.AvgPool2d(2),

            nn.Conv2d(6, 16, kernel_size=5),
            act,
            nn.AvgPool2d(2),

            nn.Conv2d(16, 120, kernel_size=5),
            act
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            act,
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, activation: str):
        super().__init__()
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "sigmoid":
            act_layer = nn.Sigmoid()
        elif activation == "tanh":
            act_layer = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_layer,
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================
#   TRAINING / EVAL
# ============================

def train_for_epochs(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     epochs: int) -> Tuple[List[float], List[float]]:
    """
    Train model for `epochs` and return (train_f1_per_epoch, val_f1_per_epoch).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    train_f1s = []
    val_f1s = []

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Evaluate train & val F1
        train_f1 = evaluate_f1(model, train_loader)
        val_f1 = evaluate_f1(model, val_loader)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

    return train_f1s, val_f1s


def evaluate_f1(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().tolist())

    return f1_score(y_true, y_pred, average="macro")


# ============================
#   GENETIC ALGORITHM
# ============================

class Individual:
    def __init__(self, batch_idx: int, act_idx: int):
        self.batch_idx = batch_idx
        self.act_idx = act_idx
        self.age = 0
        self.fitness = None  # type: float | None

    @property
    def batch_size(self) -> int:
        return BATCH_SIZES[self.batch_idx]

    @property
    def activation(self) -> str:
        return ACTIVATIONS[self.act_idx]

    def genome(self) -> Tuple[int, int]:
        return (self.batch_idx, self.act_idx)

    def copy(self):
        new = Individual(self.batch_idx, self.act_idx)
        new.age = self.age
        new.fitness = self.fitness
        return new


def init_population(pop_size: int) -> List[Individual]:
    return [
        Individual(
            batch_idx=random.randrange(len(BATCH_SIZES)),
            act_idx=random.randrange(len(ACTIVATIONS)),
        )
        for _ in range(pop_size)
    ]


def compute_fitness(ind: Individual,
                    train_ds: Dataset,
                    val_ds: Dataset) -> float:
    """
    Fitness: validation macro-F1 after FITNESS_EPOCHS of training.
    """
    batch_size = ind.batch_size
    activation = ind.activation

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, activation).to(DEVICE)
    _, val_f1s = train_for_epochs(model, train_loader, val_loader, FITNESS_EPOCHS)

    fitness = val_f1s[-1]
    ind.fitness = fitness
    return fitness


def roulette_select(population: List[Individual]) -> Individual:
    # Handle non-positive fitness by shifting
    fitnesses = [ind.fitness if ind.fitness is not None else 0.0 for ind in population]
    min_f = min(fitnesses)
    if min_f < 0:
        fitnesses = [f - min_f + 1e-6 for f in fitnesses]

    total_f = sum(fitnesses)
    if total_f == 0:
        return random.choice(population)

    r = random.random() * total_f
    cum = 0.0
    for ind, f in zip(population, fitnesses):
        cum += f
        if cum >= r:
            return ind
    return population[-1]


def one_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """
    Genome is length 2 (batch_idx, act_idx).
    One-point crossover can only cut between gene 0 and 1.
    """
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    g1 = list(parent1.genome())
    g2 = list(parent2.genome())

    # Single crossover point between 0 and len(genome)
    # For 2 genes, the only non-trivial point is 1.
    point = 1

    child1_genome = g1[:point] + g2[point:]
    child2_genome = g2[:point] + g1[point:]

    c1 = Individual(batch_idx=child1_genome[0], act_idx=child1_genome[1])
    c2 = Individual(batch_idx=child2_genome[0], act_idx=child2_genome[1])
    return c1, c2


def mutate(ind: Individual):
    if random.random() < MUTATION_RATE:
        # Mutate batch_idx
        ind.batch_idx = random.randrange(len(BATCH_SIZES))
    if random.random() < MUTATION_RATE:
        # Mutate act_idx
        ind.act_idx = random.randrange(len(ACTIVATIONS))


def age_based_selection(population: List[Individual],
                        offspring: List[Individual],
                        pop_size: int) -> List[Individual]:
    """
    Combine population and offspring, then select `pop_size` individuals
    preferring younger (age) and better fitness.
    """
    combined = population + offspring

    # Drop individuals older than MAX_AGE
    combined = [ind for ind in combined if ind.age <= MAX_AGE]

    # In case we drop too many, keep at least something
    if len(combined) < pop_size:
        combined = sorted(population + offspring, key=lambda ind: ind.age)[:pop_size]

    # Sort by age (younger first), then by fitness (higher first)
    combined = sorted(
        combined,
        key=lambda ind: (ind.age, -(ind.fitness if ind.fitness is not None else -1e9)),
    )

    return combined[:pop_size]


def run_genetic_algorithm(train_ds: Dataset, val_ds: Dataset) -> Tuple[Individual, List[float], List[float]]:
    """
    Runs GA and returns:
      - best individual from last generation
      - list of average fitness per generation
      - list of best fitness per generation
    """
    population = init_population(POP_SIZE)

    avg_fitnesses = []
    best_fitnesses = []

    for gen in range(NUM_GENERATIONS):
        print(f"\n=== Generation {gen} ===")

        # Compute fitness for individuals without fitness
        for ind in population:
            if ind.fitness is None:
                f = compute_fitness(ind, train_ds, val_ds)
                print(f"  Individual (B={ind.batch_size}, act={ind.activation}) -> fitness={f:.4f}")

        # Record stats
        fitness_values = [ind.fitness for ind in population]
        avg_f = float(np.mean(fitness_values))
        best_f = float(np.max(fitness_values))
        avg_fitnesses.append(avg_f)
        best_fitnesses.append(best_f)

        best_ind = max(population, key=lambda ind: ind.fitness)
        print(f"  Avg fitness: {avg_f:.4f}, Best fitness: {best_f:.4f}")
        print(f"  Best so far: B={best_ind.batch_size}, act={best_ind.activation}, F1={best_ind.fitness:.4f}")

        # Age everyone
        for ind in population:
            ind.age += 1

        # Generate offspring via roulette + crossover + mutation
        offspring: List[Individual] = []
        while len(offspring) < POP_SIZE:
            p1 = roulette_select(population)
            p2 = roulette_select(population)
            c1, c2 = one_point_crossover(p1, p2)
            mutate(c1)
            mutate(c2)
            offspring.append(c1)
            if len(offspring) < POP_SIZE:
                offspring.append(c2)

        # Compute fitness for offspring
        for ind in offspring:
            f = compute_fitness(ind, train_ds, val_ds)
            print(f"  Offspring (B={ind.batch_size}, act={ind.activation}) -> fitness={f:.4f}")

        # Age-based selection for next generation
        population = age_based_selection(population, offspring, POP_SIZE)

    # After all generations, pick best from last population
    best_ind_last = max(population, key=lambda ind: ind.fitness)
    print("\n=== GA Finished ===")
    print(f"Best in last generation: B={best_ind_last.batch_size}, act={best_ind_last.activation}, "
          f"F1={best_ind_last.fitness:.4f}")

    return best_ind_last, avg_fitnesses, best_fitnesses


# ============================
#   PLOTTING
# ============================

def plot_ga_fitness(avg_fitnesses: List[float], best_fitnesses: List[float]):
    generations = list(range(len(avg_fitnesses)))
    plt.figure()
    plt.plot(generations, avg_fitnesses, label="Average Fitness")
    plt.plot(generations, best_fitnesses, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Validation Macro F1")
    plt.title("Genetic Algorithm Fitness over Generations")
    plt.legend()
    plt.grid(True)
    out_path = RESULTS_DIR / "ga_fitness.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved GA fitness plot to: {out_path}")


def plot_train_f1(train_f1s: List[float]):
    epochs = list(range(1, len(train_f1s) + 1))
    plt.figure()
    plt.plot(epochs, train_f1s, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Macro F1")
    plt.title("Training F1 for Best Hyperparameters")
    plt.grid(True)
    out_path = RESULTS_DIR / "train_f1_best.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training F1 plot to: {out_path}")


# ============================
#   MAIN
# ============================
def debug_single_run_lenet(train_ds, val_ds,
                           batch_size=64,
                           lr=0.01,
                           epochs=20):
    print("\n=== Debug single run with LeNet5 ===")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = LeNet5(n_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

    train_f1s = []
    val_f1s   = []

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)              # xb shape: (B, 784)
            yb = yb.to(DEVICE)

            # 🔑 reshape flat images -> 1x28x28 for conv net
            xb = xb.view(-1, 1, 28, 28)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # --- Evaluate F1 on train & val ---
        train_f1 = evaluate_f1_lenet(model, train_loader)
        val_f1   = evaluate_f1_lenet(model, val_loader)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        print(f"Epoch {ep+1:02d}: train F1={train_f1:.4f}, val F1={val_f1:.4f}")

    print("Final train F1:", train_f1s[-1])
    print("Final val   F1:", val_f1s[-1])
    return train_f1s, val_f1s


def evaluate_f1_lenet(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            xb = xb.view(-1, 1, 28, 28)   # reshape flat → image
            logits = model(xb)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().tolist())

    return f1_score(y_true, y_pred, average="macro")
def evaluate_f1_lenet(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate macro-F1 for a conv net expecting (N,1,28,28)."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            xb = xb.view(-1, 1, 28, 28)   # flat -> image
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().tolist())
    return f1_score(y_true, y_pred, average="macro")


def train_for_epochs_lenet(model: nn.Module,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           epochs: int) -> Tuple[List[float], List[float]]:
    """
    Same as train_for_epochs, but reshapes inputs for LeNet5 (conv).
    Returns (train_f1_per_epoch, val_f1_per_epoch).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    train_f1s = []
    val_f1s = []

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            xb = xb.view(-1, 1, 28, 28)  # 🔑 flat -> 1x28x28

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Evaluate train & val F1 with LeNet-aware eval
        train_f1 = evaluate_f1_lenet(model, train_loader)
        val_f1 = evaluate_f1_lenet(model, val_loader)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

    return train_f1s, val_f1s
def overfit_tiny_subset(train_ds,
                        batch_size=64,
                        lr=0.01,
                        epochs=50):
    print("\n=== Overfit tiny subset with LeNet5 ===")

    X_all, y_all = train_ds.tensors
    n_tiny = min(256, X_all.shape[0])
    X_tiny = X_all[:n_tiny].clone()
    y_tiny = y_all[:n_tiny].clone()

    tiny_ds = TensorDataset(X_tiny, y_tiny)
    tiny_loader = DataLoader(tiny_ds, batch_size=batch_size, shuffle=True)

    model = LeNet5(n_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

    def tiny_f1():
        model.eval()
        from sklearn.metrics import f1_score
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in tiny_loader:
                xb = xb.to(DEVICE)
                xb = xb.view(-1, 1, 28, 28)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(yb.numpy().tolist())
        return f1_score(y_true, y_pred, average="macro")

    for ep in range(epochs):
        model.train()
        for xb, yb in tiny_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            xb = xb.view(-1, 1, 28, 28)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        f1 = tiny_f1()
        print(f"Epoch {ep+1:02d}: tiny-set F1={f1:.4f}")
    
def main():
    set_seed(123)

    # Load data and split
    train_ds, val_ds, test_ds = load_data_and_split()

    # Run GA to find best hyperparameters
    best_ind, avg_fit, best_fit = run_genetic_algorithm(train_ds, val_ds)

    # Plot GA fitness curves
    plot_ga_fitness(avg_fit, best_fit)

    # Retrain best model on train+val combined, then evaluate on test
    full_X = torch.cat([train_ds.tensors[0], val_ds.tensors[0]], dim=0)
    full_y = torch.cat([train_ds.tensors[1], val_ds.tensors[1]], dim=0)
    full_ds = TensorDataset(full_X, full_y)

    batch_size = best_ind.batch_size
    activation = best_ind.activation

    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, activation).to(DEVICE)
    train_f1s, _ = train_for_epochs(model, full_loader, test_loader, FINAL_TRAIN_EPOCHS)
    plot_train_f1(train_f1s)

    # Final evaluation on independent test set
    test_f1 = evaluate_f1(model, test_loader)
    print("\n=== Final Evaluation on Test Set ===")
    print(f"Best hyperparameters: batch_size={batch_size}, activation={activation}")
    print(f"Test macro F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()