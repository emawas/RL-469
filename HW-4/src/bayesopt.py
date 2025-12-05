#!/usr/bin/env python3
"""
Bayesian Optimization for EMNIST (digits only).

We reuse:
  - Data loading
  - SimpleMLP architecture
  - Training/eval utilities

from your existing genetic_tuning_emnist.py.

Search space:
  - batch size index in [0, len(BATCH_SIZES)-1]
  - activation index in [0, len(ACTIVATIONS)-1]

The black-box function returns the validation macro-F1 after a
fixed number of training epochs.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization

# 👇 import everything we need from your GA script
from genetic_tuning_emnist import (
    set_seed,
    load_data_and_split,
    SimpleMLP,
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    BATCH_SIZES,
    ACTIVATIONS,
    DEVICE,
    RESULTS_DIR,
    train_for_epochs,
    evaluate_f1,
)

# ============================
#   CONFIG FOR BO
# ============================

FITNESS_EPOCHS_BO = 15      # epochs per BO evaluation (black-box)
FINAL_TRAIN_EPOCHS_BO = 30  # epochs for final model with best hyperparams

GLOBAL_DATASETS = None      # (train_ds, val_ds, test_ds)


# ============================
#   BLACK-BOX OBJECTIVE
# ============================

def bo_objective(batch_idx_cont: float, act_idx_cont: float) -> float:
    """
    Bayesian Optimization black-box function.

    Inputs (continuous):
        batch_idx_cont \in [0, len(BATCH_SIZES)-1]
        act_idx_cont   \in [0, len(ACTIVATIONS)-1]

    We round/clamp them to integers, map to real hyperparams,
    train SimpleMLP for FITNESS_EPOCHS_BO, and return validation macro-F1.
    """
    # Map continuous -> discrete indices
    batch_idx = int(round(batch_idx_cont))
    act_idx = int(round(act_idx_cont))

    batch_idx = max(0, min(batch_idx, len(BATCH_SIZES) - 1))
    act_idx = max(0, min(act_idx, len(ACTIVATIONS) - 1))

    batch_size = BATCH_SIZES[batch_idx]
    activation = ACTIVATIONS[act_idx]

    train_ds, val_ds, _ = GLOBAL_DATASETS

    print(f"[BO] Evaluating batch_size={batch_size}, activation={activation} "
          f"(batch_idx={batch_idx}, act_idx={act_idx})")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, activation).to(DEVICE)
    _, val_f1s = train_for_epochs(model, train_loader, val_loader, FITNESS_EPOCHS_BO)

    score = float(val_f1s[-1])
    print(f"[BO] -> val F1 = {score:.4f}")
    return score


# ============================
#   MAIN BO PIPELINE
# ============================

def main():
    global GLOBAL_DATASETS

    set_seed(123)

    # 1) Load data once, reuse inside black-box
    train_ds, val_ds, test_ds = load_data_and_split()
    GLOBAL_DATASETS = (train_ds, val_ds, test_ds)

    # 2) Define search bounds over indices (continuous; we'll round inside)
    pbounds = {
        "batch_idx_cont": (0, len(BATCH_SIZES) - 1),
        "act_idx_cont":   (0, len(ACTIVATIONS) - 1),
    }

    optimizer = BayesianOptimization(
        f=bo_objective,
        pbounds=pbounds,
        random_state=123,
        verbose=2,  # 0: silent, 1: only status, 2: full
    )

    # 3) Run BO: a few random points + iterative improvement
    optimizer.maximize(
        init_points=5,   # random explorations
        n_iter=15,       # BO iterations
    )

    print("\n=== Bayesian Optimization Finished ===")
    print("Raw optimizer.max:")
    print(optimizer.max)

    # 4) Extract best discrete hyperparameters
    best_params = optimizer.max["params"]
    best_batch_idx = int(round(best_params["batch_idx_cont"]))
    best_act_idx   = int(round(best_params["act_idx_cont"]))

    best_batch_idx = max(0, min(best_batch_idx, len(BATCH_SIZES) - 1))
    best_act_idx   = max(0, min(best_act_idx, len(ACTIVATIONS) - 1))

    best_B   = BATCH_SIZES[best_batch_idx]
    best_act = ACTIVATIONS[best_act_idx]

    print(f"\nBest hyperparameters from Bayesian Optimization:")
    print(f"  batch_size index: {best_batch_idx} -> {best_B}")
    print(f"  activation index: {best_act_idx}   -> {best_act}")

    # 5) Retrain final model on (train + val) with best hyperparams
    full_X = torch.cat([train_ds.tensors[0], val_ds.tensors[0]], dim=0)
    full_y = torch.cat([train_ds.tensors[1], val_ds.tensors[1]], dim=0)
    full_ds = TensorDataset(full_X, full_y)

    full_loader = DataLoader(full_ds, batch_size=best_B, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=best_B, shuffle=False)

    final_model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, best_act).to(DEVICE)
    train_f1s, _ = train_for_epochs(final_model, full_loader, test_loader, FINAL_TRAIN_EPOCHS_BO)

    # 6) Plot training F1 vs epoch for BO-chosen hyperparams
    epochs = list(range(1, len(train_f1s) + 1))
    plt.figure()
    plt.plot(epochs, train_f1s, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Macro F1")
    plt.title("Training F1 for Best Hyperparameters (Bayesian Optimization)")
    plt.grid(True)

    out_path = RESULTS_DIR / "train_f1_best_bo.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved BO training F1 plot to: {out_path}")

    # 7) Final evaluation on independent test set
    test_f1 = evaluate_f1(final_model, test_loader)
    print("\n=== Final Evaluation on Test Set (Bayesian Optimization) ===")
    print(f"Best hyperparameters from BO: batch_size={best_B}, activation={best_act}")
    print(f"Test macro F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()