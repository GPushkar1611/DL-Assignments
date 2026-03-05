# Deep Learning 2026 — Assignment 1

## Repository Structure

```
ASSIGNMENT-1/
├── adaline/
│   ├── adaline.py
│   └── run_adaline_experiment.py
├── mlp/
│   ├── activations.py
│   ├── experiements.ipynb
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

## Solutions

### Q1 — Artificial Neural Network with Backpropagation *(20 Marks)*
**Solution:** [`ANN.ipynb`](./ANN.ipynb)

ANN implemented from scratch with backpropagation to learn XOR and Cosine functions (Gaussian-noised inputs, 80/20 train-test split). Includes loss/accuracy plots, and experiments with 3 values of `n` (Deterministic GD) and 3 batch sizes `m` (Stochastic GD).

---

### Q2 — Universal Approximation Theorem *(15 Marks)*
**Solution:** [`Assignment_Report.pdf`](./Assignment_Report.pdf)

Covers: formal UAT statement (function class, domain, approximation norm), proof outline via Stone–Weierstrass, two practical limitations, and UAT behaviour under ReLU with increasing depth.

---

### Q3 — One Step of Gradient Descent in a Deep Neural Network *(25 Marks)*
**Solution:** [`Assignment_Report.pdf`](./Assignment_Report.pdf)

Manual forward pass, full backpropagation, and one gradient descent update for a 3-layer ReLU network under two loss functions:
- **Part A** — Softmax + Cross-Entropy Loss (forward pass, L_CE, gradients w.r.t. logits, all weight/bias gradients, updated parameters)
- **Part B** — Multiclass Hinge Loss (hinge loss, gradients w.r.t. logits, all weight/bias gradients, updated parameters)

---

### Q4 — A Voyage into Neural Networks: Dataset Construction
**Dataset Generation:** [`data_generation.py`](./data_generation.py)
**Dataset Description:** [`dataset_description.pdf`](./dataset_description.pdf)
**Generated Dataset:** [`iit_h_mess_dataset.csv`](./iit_h_mess_dataset.csv)

500-sample tabular dataset simulating IIT-H student mess duration, extended to 5000 samples via Gaussian noise augmentation on `weather_features`, `rising_time`, and `sleeping_time`.

---

### Q5 — ADALINE (Adaptive Linear Neuron) *(40 Points)*

**5.1 — Theory (20 pts):** [`Assignment_Report.pdf`](./Assignment_Report.pdf)
ADALINE learning algorithm (Widrow & Hoff, 1960) and convergence proof to the global least-squares solution.

**5.2 — Implementation (10 pts):** [`adaline/adaline.py`](./adaline/adaline.py)
`Adaline` class with `fit()`, `predict()`, and `score()` methods.

**5.3 — Experiments (10 pts):** [`adaline/run_adaline_experiment.py`](./adaline/run_adaline_experiment.py)
Dataset plot, MSE vs. epoch, 2D decision boundary (PCA), train/val error curves, learning rate sweep η ∈ {0.01, 0.1, 1.0, 10.0}, and final accuracy vs. training set size (10%–100%). Plots saved to [`plots/adaline-experiment/`](./plots/adaline-experiment/).

---

### Q6 — Multi-Layer Perceptron with Backpropagation

**6.1 — Theory (10 pts):** [`Assignment_Report.pdf`](./Assignment_Report.pdf)
Vanishing gradient problem (mathematical derivation for sigmoid) and effect of random dropout (p=0.5) on the chain rule.

**6.2 — Implementation (15 pts):**
- [`mlp/mlp.py`](./mlp/mlp.py) — `MLP` class with `forward()`, `backward()`, and `fit()` with early stopping
- [`mlp/activations.py`](./mlp/activations.py) — Sigmoid, Tanh, ReLU, Leaky ReLU and derivatives
- [`mlp/optimizers.py`](./mlp/optimizers.py) — SGD, Momentum, Adam, Nesterov-AG, AdaGrad, RMSProp, Muon; Xavier-Glorot & He weight initialisation

**6.3 — Architecture Ablation (15 pts):** [`mlp/experiements.ipynb`](./mlp/experiements.ipynb)
- *Depth Ablation* — fixed width 64, depths 1–4; train/val curves, test accuracy, training time, overfitting analysis, test accuracy vs. depth summary plot
- *Width Ablation* — fixed depth 2, widths 8→256; test accuracy vs. parameter count (log scale)
- *Activation Comparison* — Sigmoid, Tanh, ReLU, Leaky ReLU on [input, 64, 64, output]; loss curves, gradient statistics, dead neuron analysis, per-layer gradient magnitude (sigmoid vs. ReLU)

**6.4 — Loss Function Analysis (10 pts):** [`mlp/experiements.ipynb`](./mlp/experiements.ipynb)
Three loss functions compared on the mess duration task with discussion on suitability.

**6.5 — Optimizer Comparison (10 pts):** [`mlp/experiements.ipynb`](./mlp/experiements.ipynb)
SGD, SGD+Momentum, Adam, AdaGrad, Nesterov-AG, RMSProp, and Muon compared on convergence curves, time to 90% validation accuracy, weight update magnitudes, and learning rate sensitivity.

**6.6 — Regularization (5 pts):** [`mlp/experiements.ipynb`](./mlp/experiements.ipynb)
L1 and L2 regularization (λ ∈ {0.001, 0.01, 0.1}) vs. no regularization; train/val accuracy curves and L1 weight sparsity vs. λ.

**6.7 — Success Analysis (5 pts):** [`mlp/experiements.ipynb`](./mlp/experiements.ipynb)
- PCA projection of penultimate layer activations coloured by binned mess duration class
- MLP [input, 64, 64, output] vs. ADALINE comparison on test accuracy (±5 min), MSE, and training time
- Analysis and discussion of results in [`Assignment_Report.pdf`](./Assignment_Report.pdf)

Plots saved to [`plots/mlp-experiment/`](./plots/mlp-experiment/).

---

### Q7 — Neural Network Features for Kernel Methods *(Advanced)*

**7.1 — Theory (10 pts):** [`Assignment_Report.pdf`](./Assignment_Report.pdf)
Valid kernel functions, Mercer's theorem, and the interpretation of hidden layers as learned feature maps φ(x) in relation to the kernel trick.

**7.2 — Implementation (20 pts):** [`Kernel_methods.ipynb`](./Kernel_methods.ipynb)
- *Neural Feature Extractor* — penultimate layer features φ_NN(x) = h^(L−1)(x) from best MLP, t-SNE visualisation, neural kernel SVM, and kernel SVR on Phase 4 data
- *Standard Kernel Comparison* — Linear, Polynomial (d ∈ {2,3}), RBF (γ ∈ {0.01, 0.1, 1.0}), and Neural kernels compared on test accuracy, 2D decision boundaries, and kernel matrix K_ij visualisation