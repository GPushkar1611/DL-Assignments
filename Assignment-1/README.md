# Deep Learning 2026 — Assignment 1

## Repository Structure
```
ASSIGNMENT-1/
├── adaline/
│   ├── adaline.py
│   └── run_adaline_experiment.py
├── mlp/
│   ├── activations.py
│   ├── experiments.ipynb
│   ├── mlp.py
│   └── optimizers.py
├── plots/
│   ├── adaline-experiment/
│   └── mlp-experiment/
├── ANN.ipynb
├── Assignment_Report.pdf
├── data_generation.py
├── dataset_description.pdf
├── iit_h_mess_dataset.csv
├── Kernel_methods.ipynb
└── README.md
```

---

## Setup & Configuration

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2. Set your plot output directory

Before running any experiment script, open it and update the `PLOT_DIR` variable to point to your local plots folder.

**In `adaline/run_adaline_experiment.py`** — set it once at the top:
```python
PLOT_DIR = "plots/adaline-experiment"   # ← change this to your path
```

**In `mlp/experiments.ipynb`** — `PLOT_DIR` is redefined at the start of each section and subsection to save plots in different subfolders. You will need to update **every occurrence** to match your local path. Search for `PLOT_DIR =` in the notebook and update each one, for example:
```python
PLOT_DIR = "plots/mlp-experiment/depth-ablation"      # ← change base path
PLOT_DIR = "plots/mlp-experiment/width-ablation"      # ← change base path
PLOT_DIR = "plots/mlp-experiment/activations"         # ← change base path
# ... and so on for each section
```

Use an absolute path if you're running from a different working directory, for example:
```python
# Windows
PLOT_DIR = r"C:\Users\YourName\DL-Assignments\Assignment-1\plots\mlp-experiment\depth-ablation"

# macOS / Linux
PLOT_DIR = "/home/yourname/DL-Assignments/Assignment-1/plots/mlp-experiment/depth-ablation"
```

All directories are created automatically if they don't exist (`os.makedirs` with `exist_ok=True`), so you only need to set the paths — no manual folder creation required.

### 3. Dataset path

All scripts assume `iit_h_mess_dataset.csv` is in the working directory from which you run them. If you run a script from inside the `adaline/` folder, you may need to adjust the path:
```python
df = pd.read_csv("../iit_h_mess_dataset.csv")   # if running from adaline/
```

---

## Running the Experiments

### ADALINE (Q5)
```bash
cd adaline
python run_adaline_experiment.py
```

Plots are saved to your configured `PLOT_DIR`.

### MLP (Q6)

Open and run `mlp/experiments.ipynb` cell by cell in Jupyter. Ensure `PLOT_DIR` is set correctly in the first cell.

### ANN (Q1)

Open and run `ANN.ipynb` in Jupyter.

### Kernel Methods (Q7)

Open and run `Kernel_methods.ipynb` in Jupyter.

---

## Solutions

### Q1 — Artificial Neural Network with Backpropagation *(20 Marks)*
**Solution:** [`ANN.ipynb`](./ANN.ipynb)

ANN implemented from scratch with backpropagation to learn XOR and Cosine functions (Gaussian-noised inputs, 80/20 train-test split). Includes loss/accuracy plots, and experiments with 3 values of $n$ (Deterministic GD) and 3 batch sizes $m$ (Stochastic GD).

---

### Q2 — Universal Approximation Theorem *(15 Marks)*
**Solution:** [`Assignment_Report.pdf`](./Assignment_Report.pdf)

Covers: formal UAT statement (function class, domain, approximation norm $\|\cdot\|_\infty$), proof outline via Stone–Weierstrass, two practical limitations, and UAT behaviour under ReLU with increasing depth.

---

### Q3 — One Step of Gradient Descent in a Deep Neural Network *(25 Marks)*
**Solution:** [`Assignment_Report.pdf`](./Assignment_Report.pdf)

Manual forward pass, full backpropagation, and one gradient descent update for a 3-layer ReLU network under two loss functions:

- **Part A — Softmax + Cross-Entropy Loss:** forward pass, $\mathcal{L}_{\text{CE}}$, gradients $\partial \mathcal{L}/\partial \mathbf{z}$, all $\nabla_{W^{(l)}}, \nabla_{\mathbf{b}^{(l)}}$, and updated parameters
- **Part B — Multiclass Hinge Loss:** $\mathcal{L}_{\text{hinge}}$, gradients $\partial \mathcal{L}/\partial \mathbf{z}$, all $\nabla_{W^{(l)}}, \nabla_{\mathbf{b}^{(l)}}$, and updated parameters

---

### Q4 — Dataset Construction

**Dataset Generation:** [`data_generation.py`](./data_generation.py)
**Dataset Description:** [`dataset_description.pdf`](./dataset_description.pdf)
**Generated Dataset:** [`iit_h_mess_dataset.csv`](./iit_h_mess_dataset.csv)

500-sample tabular dataset simulating IIT-H student mess duration, extended to 5000 samples via Gaussian noise augmentation on `weather_features`, `rising_time`, and `sleeping_time`.

---

### Q5 — ADALINE *(40 Points)*

**5.1 — Theory (20 pts):** [`Assignment_Report.pdf`](./Assignment_Report.pdf)
ADALINE learning rule (Widrow & Hoff, 1960) and convergence proof to the global least-squares solution $\mathbf{w}^* = \arg\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$.

**5.2 — Implementation (10 pts):** [`adaline/adaline.py`](./adaline/adaline.py)
`Adaline` class with `fit()`, `predict()`, and `score()` methods.

**5.3 — Experiments (10 pts):** [`adaline/run_adaline_experiment.py`](./adaline/run_adaline_experiment.py)
Dataset plot, MSE vs. epoch, 2D decision boundary (PCA), train/val error curves, learning rate sweep $\eta \in \{0.01, 0.1, 1.0, 10.0\}$, and final validation loss vs. training set size (10%–100%). Plots saved to [`plots/adaline-experiment/`](./plots/adaline-experiment/).

---

### Q6 — Multi-Layer Perceptron with Backpropagation

**6.1 — Theory (10 pts):** [`Assignment_Report.pdf`](./Assignment_Report.pdf)
Vanishing gradient problem (mathematical derivation for sigmoid: $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$) and effect of random dropout ($p = 0.5$) on the chain rule.

**6.2 — Implementation (15 pts):**
- [`mlp/mlp.py`](./mlp/mlp.py) — `MLP` class with `forward()`, `backward()`, and `fit()` with early stopping
- [`mlp/activations.py`](./mlp/activations.py) — Sigmoid, Tanh, ReLU, Leaky ReLU and their derivatives
- [`mlp/optimizers.py`](./mlp/optimizers.py) — SGD, Momentum, Adam, Nesterov-AG, AdaGrad, RMSProp, Muon; Xavier-Glorot & He weight initialisation

**6.3 — Architecture Ablation (15 pts):** [`mlp/experiments.ipynb`](./mlp/experiments.ipynb)
- *Depth Ablation* — fixed width 64, depths $L \in \{1, 2, 3, 4\}$
- *Width Ablation* — fixed depth 2, widths $d \in \{8, \ldots, 256\}$; test loss vs. parameter count (log scale)
- *Activation Comparison* — Sigmoid, Tanh, ReLU, Leaky ReLU on $[\text{input}, 64, 64, \text{output}]$

**6.4 — Loss Function Analysis (10 pts):** [`mlp/experiments.ipynb`](./mlp/experiments.ipynb)
Three loss functions compared on the mess duration task with discussion on suitability.

**6.5 — Optimizer Comparison (10 pts):** [`mlp/experiments.ipynb`](./mlp/experiments.ipynb)
SGD, SGD+Momentum, Adam, AdaGrad, Nesterov-AG, RMSProp, and Muon compared on convergence, time to near-best validation loss, weight update magnitudes $\|\Delta W\|$, and learning rate sensitivity.

**6.6 — Regularization (5 pts):** [`mlp/experiments.ipynb`](./mlp/experiments.ipynb)
$L_1$ and $L_2$ regularization with $\lambda \in \{0.001, 0.01, 0.1\}$ vs. no regularization.

**6.7 — Success Analysis (5 pts):** [`mlp/experiments.ipynb`](./mlp/experiments.ipynb)
PCA projection of penultimate-layer activations, MLP vs. ADALINE comparison on test MSE and training time.

Plots saved to [`plots/mlp-experiment/`](./plots/mlp-experiment/).

---

### Q7 — Neural Network Features for Kernel Methods *(Advanced)*

**7.1 — Theory (10 pts):** [`Assignment_Report.pdf`](./Assignment_Report.pdf)
Valid kernel functions, Mercer's theorem, and hidden layers as learned feature maps $\phi(x)$ in relation to the kernel trick $k(x, x') = \langle \phi(x), \phi(x') \rangle$.

**7.2 — Implementation (20 pts):** [`Kernel_methods.ipynb`](./Kernel_methods.ipynb)
- *Neural Feature Extractor* — penultimate-layer features $\phi_{\text{NN}}(x) = h^{(L-1)}(x)$, $t$-SNE visualisation, neural kernel SVM and SVR
- *Standard Kernel Comparison* — Linear, Polynomial ($d \in \{2,3\}$), RBF ($\gamma \in \{0.01, 0.1, 1.0\}$), and Neural kernels compared on test accuracy, decision boundaries, and kernel matrix $K_{ij}$ visualisation