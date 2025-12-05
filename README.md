# RL-469

# HW-4 Instructions on How to Run the Code  
Genetic Algorithm + Bayesian Optimization for EMNIST (Digits Only)  
RL-469 — Assignment 4

## Clone the RL-469 repository
```
git clone https://github.com/emawas/RL-469.git
```


Then navigate to the HW-4 directory:

```
cd RL-469/HW-4
```

This homework implements two AutoML methods for tuning:

- **Mini-batch size B**
- **Activation function** ∈ {relu, sigmoid, tanh}

The dataset used is EMNIST digits (10 classes) loaded from:


```
HW-4/Assignment3-data/
```
# Part 1 — Genetic Algorithm (GA)

## Running the Genetic Algorithm

The GA implementation is located in:
```
HW-4/src/genetic_tuning_emnist.py
```

To run the full GA pipeline:


nohup python -u src/genetic_tuning_emnist.py > results/ga_run.log 2>&1 &
tail -f results/ga_run.log

This script:

1. Loads and splits EMNIST (train/val/test)
2. Runs the Genetic Algorithm using:
   - Roulette selection
   - One-point crossover
   - Age-based survivor selection
3. Finds the best hyperparameters (batch size + activation)
4. Retrains on train+validation
5. Evaluates on test set
6. Saves plots

### Output files

GA fitness across generations:

```
HW-4/results/ga_fitness.png
```

Training F1 curve of best GA model:

```
HW-4/results/train_f1_best.png
```
GA run logs:
```
HW-4/results/ga_run.log
```
# Part 2 — Bayesian Optimization (BO)

## Running Bayesian Optimization

The BO implementation is located in:

```
HW-4/src/bayesopt.py
```


## HW 3 Instructions on how to run the code:

## Part 1

### Clone RL-469 repo 

```
git clone https://github.com/emawas/RL-469.git
```
To run the sequential FedAvg problem in question 1. 

```
cd RL-469/HW-3
```

Then go to the bottom under 
```
if __name__ == "__main__":
```
you will find a config array that allows you to modify C and E 

```
configs = [
        (0.10, 10)
    ]
```

Then in th terminal you run:

```
nohup python -u src/train_fedav.py > results/fedavg_C0.1_E10.log 2>&1 &
tail -f results/fedavg_C0.1_E10.log 
```
The results of the test run will show after the training is done and the results can be found in 

```
HW-3/results/C0.1_E10/live_plot.png
```

And the run on the test data will show as a png file named 

```
HW-3\results\best_C0.1_E10.png
```

## Parallel FedAvg with Ray

Now to run the parallel fedav script with ray we run the following:

```
cd HW-3
```
Near the bottom of src/train_fedav_ray.py, under:

```
if __name__ == "__main__":
```

You will find the configuration llist that controls the fraction of participating clients C and the number of local epochs E:

```
configs = [
    (0.10, 10)
]
```

The best parameters and hyperparameters are already chosen. Then in the terminal run:

```
nohup python -u src/train_fedav_ray.py > results/fedavg_ray_C0.1_E10.log 2>&1 &
tail -f results/fedavg_ray_C0.1_E10.log 
```

You will find the results in:

```
HW-3/results/ray_C0.1_E10.png
```

## Laplace noise Part 3

Run the following 

```
nohup python -u src/dp_fedav.py > results/dp_fedav.log 2>&1 &
tail -f results/dp_fedav.log
```
That will run the model with different values of b and the corresponding accuracy vs. b plot can be found in 

```
HW-3\results\dp_accuracy_vs_b.png
```


# HW 1 REPORT
In this project, we train agents to solve control tasks using **policy gradient methods**:

- **Without a baseline**: the policy directly uses Monte Carlo returns as targets (high variance).
- **With a baseline**: a learned value function (`V(s)`) approximates expected returns, and the *advantage* `A(s,a) = G_t - V(s)` replaces raw returns to reduce variance.

A lower learning rate (`lr = 0.001`) was chosen empirically because higher rates (e.g. `0.01`) caused unstable gradients and stalled learning in both environments. With `0.001`, training was stable and monotonic.
```text
RL-469/
└── HW-1/
    ├── src/
    │   ├── cartpole/
    │   │   ├── train_cartpole_pg.py              # Vanilla REINFORCE for CartPole
    │   │   ├── train_cartpole_pg_baseline.py     # REINFORCE + Value Baseline (Actor–Critic)
    │   │   ├── rollout_cartpole_pg.py            # Evaluate trained CartPole policy
    │   │   ├── policy_mlp.py                     # Policy (MLP)
    │   │   └── value_mlp.py                      # Value network (critic)
    │   │
    │   ├── pong/
    │   │   ├── train_pong_pg.py                  # Vanilla REINFORCE for Pong
    │   │   ├── train_pong_pg_baseline.py         # REINFORCE + Value Baseline for Pong
    │   │   ├── rollout_pong_pg.py                # Evaluate Pong policy
    │   │   └── policy_cnn.py                     # CNN-based policy for Atari frame
    │   │   └── preprocess.py                     # Frame preprocessing and helpers
    │   │   └── value_cnn.py                      # Value network (critic) CNN
    │   │
    ├── results/                                  # Training logs, models, and plots
    │   ├── cartpole_pg_YYYYMMDD_HHMMSS/
    │   ├── cartpole_pg_baseline_YYYYMMDD_HHMMSS/
    │   ├── pong_gpu_YYYYMMDD_HHMMSS/
    │   └── pong_pg_baseline_gpu_YYYYMMDD_HHMMSS/
    │
    └── README.md
```
## Cartpole (no baseline)
We start by training a model that had no baseline and just follows the vanilla REINFORCE algorithm
### 1. Training
 - Create and activate environment
```
conda create -n HW1 python=3.13 -y
conda activate HW1
```
 - Install dependencies
```
pip install torch torchvision torchaudio
pip install gymnasium[atari,accept-rom-license] ale-py matplotlib pandas
```
 - Verify GPU access
```
python -c "import torch; print(torch.cuda.is_available())"
```
Each experiment automatically creates a folder under `results/` containing:
- `rewards.csv` — episode returns and loss
- `policy.pt` — trained policy checkpoint
- `live_learning_curve.png` — generated during training if live plotter is running
- optional `train.log` — live output when running in background
to train we run teh folowing in the bash:
```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.cartpole.train_cartpole_pg \
  --episodes 1500 \
  --gamma 0.95 \
  --lr 0.001 \
  --hidden 64 \
  --seed 42 \
  --results_dir results/cartpole_pg_$(date +%Y%m%d_%H%M%S) \
  > train.log 2>&1 &
```
The results can be found in: results/cartpole_pg_<timestamp>/
<img width="960" height="720" alt="image" src="https://github.com/user-attachments/assets/42a97b8f-c36e-4d37-8a60-553ebc58d564" />

We see that without a baseline, the returns are very noisy. Hence, we use a baseline to reduce the variance.
### Cartpole with a baseline

Vanilla REINFORCE uses the return-to-go $\(G_t\)$ to weight the log-probability gradients:

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\\left[  \nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t\right].$$

This estimator has high variance. We reduce variance by subtracting a **baseline** $\(b(s_t)\)$ that does not depend on the action:

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\; \big(G_t - b(s_t)\big)\right].$$

The optimal (MSE) baseline is the state value $\(V^\pi(s_t) = \mathbb{E}[G_t \mid s_t]\)$.  
We learn a parametric value function $\(V_\phi(s)\)$ (the **critic**) and use it as the baseline. We approximate it with a neural network, while the policy acts as the actor. 

In our implementation:

```
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # output shape (T,)
```

#### What we actually optimize

**Actor (policy) loss — REINFORCE with baseline**

$$\mathcal{L}_{\text{actor}}=-\,\frac{1}{T} \sum_{t=0}^{T-1}\log \pi_\theta(a_t \mid s_t)\ \underbrace{\big(G_t - V_\phi(s_t)\big)}_{A_t\ \text{(advantage)}}.$$

**Critic (value) loss — fit returns with MSE**

$`\mathcal{L}_{\text{critic}}=\frac{1}{2T}\sum_{t=0}^{T-1}\big(G_t - V_\phi(s_t)\big)^2.`$
During training, we compute the return for each episode:

```
G = compute_returns(rewards, gamma)
```
The critic outpute V and we compute the advantage:

```
V = valuef(S)
advantages = G - V
```

### Training with a baseline:
To train Cartpole with a baseline we run the following commands:

```
CUDA_VISIBLE_DEVICES=0 python -u -m src.cartpole.train_cartpole_pg_baseline \
  --episodes 1500 \
  --gamma 0.95 \
  --lr_pi 0.001 \
  --lr_v 0.005 \
  --hidden 64 \
  --seed 42 \
  --results_dir results/cartpole_pg_baseline_$(date +%Y%m%d_%H%M%S) \
  > train.log 2>&1 &
```
And we get the following results:
<img width="1050" height="750" alt="image" src="https://github.com/user-attachments/assets/553b0128-de5e-40ed-9b35-87f810ddd584" />
We see that the returns have much less variance as compared to the model with no baseline and we see the following:
	•	Faster & smoother learning: The baseline (critic) keeps the policy gradients centered, so updates are less noisy.
	•	Lower variance curves: When you plot return vs. episodes, the baseline runs have visibly tighter (less jagged) curves. Rolling-average (MA100) lines will be smoother compared to the no-baseline runs.

### 2. Rollouts 

After training run the following commands to produce 500 rollout terms:

```
export RUN_CART="results/cartpole_pg_20251018_230000"
export CKPT="$RUN_CART/policy.pt"
export OUT="$RUN_CART/rollouts_500"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 python -u -m src.cartpole.rollout_cartpole_pg \
  --checkpoint "$CKPT" \
  --episodes 500 \
  --seed 123 \
  --device cuda \
  --results_dir "$OUT"
```

This produces:
	•	returns.npy — all 500 rollout returns
	•	rollout_hist.png — histogram of return distribution

  This produced the following plot 
  <img width="960" height="720" alt="image" src="https://github.com/user-attachments/assets/7f491b55-5c29-4303-a155-b0b7e4076482" />

And for the model with a baseline we run the following commands:

```
export RUN_CART_BASE="results/cartpole_pg_baseline_20251018_230000"
export CKPT="$RUN_CART_BASE/checkpoint.pt"
export OUT="$RUN_CART_BASE/rollouts_500"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 python -u -m src.cartpole.rollout_cartpole_baseline \
  --checkpoint "$CKPT" \
  --episodes 500 \
  --seed 123 \
  --device cuda \
  --results_dir "$OUT"
```
We get the following plot:
<img width="1050" height="750" alt="image" src="https://github.com/user-attachments/assets/072647e7-3703-410b-957a-6ea73cec82db" />

## Pong (no baseline)

Now we turn to training the model on a much larger environment. This model took much longer to run even on a GPU, and was nearly impossible without using a baseline due to the variance in the returns and how sparse the rewards are. We run the same REINFORCE algorithm but this time on the pong environment:

This experiment trains an Atari Pong agent using the standard REINFORCE algorithm (no critic / baseline).
All logs and model checkpoints will be saved under results/pong_gpu_YYYYMMDD_HHMMSS/.

```
# Launch training on GPU 0 (replace with your GPU index if needed)
CUDA_VISIBLE_DEVICES=0 python -u -m src.pong.train_pong_pg \
  --episodes 100000 \
  --gamma 0.99 \
  --lr 0.0001 \
  --hidden 200 \
  --seed 7 \
  --results_dir results/pong_gpu_$(date +%Y%m%d_%H%M%S) \
  > train.log 2>&1 &
```
Once training starts, you'll see a new folder in your results/directory like:

```text
results/
└── pong_gpu_20251018_200321/
    ├── rewards.csv        # Episode vs. return (used for plotting learning curves)
    ├── policy.pt          # Trained policy weights
    ├── train.log          # Console output and losses
    └── checkpoints/       # Periodic saved checkpoints (every N episodes)
```
The plot should look like this:
<img width="980" height="980" alt="image" src="https://github.com/user-attachments/assets/5348025e-57c4-4e45-85d8-2a4d217e89f9" />
We see that without a baseline, the model suffers from high variance in the returns, and training is incredibly slow.
### Note about the learning rate

During experimentation, I initially used a learning rate of 0.0001, which was the first value that produced stable and monotonic training for both environments without divergence. Higher learning rates (e.g., 0.001 or above) often caused gradient instability or flatlining early in training, particularly in the high-variance Pong environment.

Although 0.0001 led to slower convergence — with the policy improving steadily but plateauing after a few thousand episodes — it consistently avoided oscillations or crashes. Results were promising up to that plateau, but due to the time constraints of the HW assignment, there wasn’t sufficient time to perform additional sweeps with higher learning rates or more advanced variance-reduction methods (e.g., adaptive baselines or entropy regularization).

## Pong (with a baseline)
Now using the same baseline method that we used with cartpole, we can train a model with the following commands:

```
# Launch training on GPU 0 (change the device index if needed)
CUDA_VISIBLE_DEVICES=0 python -u -m src.pong.train_pong_pg_baseline \
  --episodes 100000 \
  --gamma 0.99 \
  --lr_pi 0.0001 \
  --lr_v  0.0005 \
  --hidden 200 \
  --seed 7 \
  --results_dir results/pong_pg_baseline_gpu_$(date +%Y%m%d_%H%M%S) \
  > train.log 2>&1 &
```
	•	--lr_pi is the policy (actor) LR.
	•	--lr_v is the value (critic) LR; a slightly larger LR often helps the critic track returns faster.
A new results folder will appear, e.g.:
```text
results/
└── pong_pg_baseline_gpu_20251018_235646/
    ├── rewards.csv          # episode, return, (optionally loss columns if you log them)
    ├── checkpoint.pt        # { "policy": ..., "value": ... } (latest)
    ├── checkpoints/         # periodic snapshots: ckpt_ep1000.pt, ckpt_ep2000.pt, ...
    ├── train.log            # console prints (losses, MA100, etc.)
    └── learning_curve.png   # (if your script saves one at the end)
```

And we see that training yields the following results:

<img width="980" height="980" alt="image" src="https://github.com/user-attachments/assets/c2aa8cc3-30a3-4746-bda0-42a8d832d8aa" />
Adding a value baseline made a dramatic improvement in training stability and sample efficiency.
	•	The vanilla REINFORCE agent (no baseline) quickly improved from around –21 → –14 in the first few thousand episodes, but then plateaued near –10, showing slow, noisy progress with high return variance.
	•	In contrast, the actor–critic version (with baseline) reached an average score of 0 by ~7000 episodes, effectively solving Pong while maintaining smoother, more monotonic learning curves.

To demonstrate this we run 500 rollouts (with and without a baseline using the following commands

### PONG Rollouts with no baseline:
After training, you can run evaluation rollouts to test the learned policy over 500 episodes and measure average performance.

```
# Path to your trained (no-baseline) run
export RUN_PONG="results/pong_gpu_20251018_200321"
export CKPT="$RUN_PONG/policy.pt"
export OUT="$RUN_PONG/rollouts_500"
mkdir -p "$OUT"

# Run evaluation on GPU 1 (change device index if needed)
CUDA_VISIBLE_DEVICES=1 python -u -m src.pong.rollout_pong_pg \
  --checkpoint "$CKPT" \
  --episodes 500 \
  --seed 123 \
  --device cuda \
  --results_dir "$OUT"
```

Output directory structure:
```text
results/
└── pong_gpu_20251018_200321/
    ├── rollouts_500/
    │   ├── returns.npy         # all episodic returns
    │   ├── histogram.png       # histogram of returns
    │   └── rollout_summary.txt # mean, std, and summary stats
```
### PONG Rollouts with a baseline:
We can now test the model with a baseline over 500 episodes to compare to the model without a baseline and we run witht hte following command 

```
# Path to your trained (with-baseline) run
export RUN_PONG_BASE="results/pong_pg_baseline_gpu_20251018_235646"
export CKPT="$RUN_PONG_BASE/checkpoints/ckpt_ep7000.pt"  # use the latest checkpoint
export OUT="$RUN_PONG_BASE/rollouts_500"
mkdir -p "$OUT"

# Run evaluation on GPU 1
CUDA_VISIBLE_DEVICES=1 python -u -m src.pong.rollout_pong_pg_baseline \
  --checkpoint "$CKPT" \
  --episodes 500 \
  --seed 123 \
  --device cuda \
  --results_dir "$OUT"
```
And the output directory structur will look like this:
```text
results/
└── pong_pg_baseline_gpu_20251018_235646/
    ├── rollouts_500/
    │   ├── returns.npy
    │   ├── histogram.png
    │   └── rollout_summary.txt
```
And we end with the following results:

<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/ee6d7468-4f1c-4267-936c-d981ed2f9ba9" />

<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/e5e66c2a-0d05-439b-a7ed-51aa08c5d46a" />

### Effect of the Baseline

The introduction of a value function baseline significantly improved performance stability and learning efficiency in Pong.
As shown below, the baseline version achieved a mean episode return of −0.78 with a standard deviation of 5.57, compared to the no-baseline version, which averaged −9.30 with a similar variance (Std = 4.49).

This demonstrates that adding a learned critic reduces gradient variance and stabilizes updates — allowing the agent to reach near-zero performance (roughly break-even play) within 7,000 episodes, while the vanilla REINFORCE model plateaued around −10.
In essence, the baseline allows the agent to focus on meaningful improvements rather than being overwhelmed by high-variance return estimates.
