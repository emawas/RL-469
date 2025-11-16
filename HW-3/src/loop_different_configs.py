configs = [
    (0.10, 1),
    (0.10, 5),
    (0.05, 1),
    (0.05, 5),
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
