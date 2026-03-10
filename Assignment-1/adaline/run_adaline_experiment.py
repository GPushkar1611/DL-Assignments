import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adaline import Adaline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(42)

PLOT_DIR = r"C:\Users\PUSHKAR\Desktop\DL-Assignments\Assignment-1\plots\adaline-experiment"
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("iit_h_mess_dataset.csv")

X = df.drop(columns=["mess_duration"]).values
y = df["mess_duration"].values

# ----------------------------
# Feature scaling
# ----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ----------------------------
# Train / Validation split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# PCA (fit ONLY on training)
# ----------------------------
pca = PCA(n_components=2)
pca.fit(X_train)

X_pca = pca.transform(X)
X_train_pca = pca.transform(X_train)

# ----------------------------
# Dataset visualization
# ----------------------------
plt.figure()

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    s=10,
    cmap="viridis"
)

plt.colorbar(label="Mess Duration")
plt.title("Dataset Visualization (PCA Projection)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")

plt.savefig(f"{PLOT_DIR}/dataset_pca.png")
plt.close()

# ----------------------------
# Train ADALINE
# ----------------------------
model = Adaline(learning_rate=0.01, max_iterations=1000)

train_mse, val_mse = model.fit(
    X_train,
    y_train,
    X_val,
    y_val
)

# ----------------------------
# MSE vs Epoch
# ----------------------------
plt.figure()

plt.plot(train_mse, label="Train MSE")
plt.plot(val_mse, label="Validation MSE")

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training and Validation MSE vs Epoch")

plt.legend()

plt.savefig(f"{PLOT_DIR}/mse_vs_epoch.png")
plt.close()

# ----------------------------
# Decision Function in PCA space
# ----------------------------
xx, yy = np.meshgrid(
    np.linspace(X_train_pca[:,0].min(), X_train_pca[:,0].max(), 100),
    np.linspace(X_train_pca[:,1].min(), X_train_pca[:,1].max(), 100)
)

grid = np.c_[xx.ravel(), yy.ravel()]

grid_original = pca.inverse_transform(grid)

zz = model.predict(grid_original).reshape(xx.shape)

plt.figure()

plt.contourf(xx, yy, zz, levels=30, cmap="viridis")

plt.scatter(
    X_train_pca[:,0],
    X_train_pca[:,1],
    c=y_train,
    s=10
)

plt.title("ADALINE Decision Function (PCA Space)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")

plt.savefig(f"{PLOT_DIR}/decision_boundary.png")
plt.close()

# ----------------------------
# Learning rate experiment
# ----------------------------
learning_rates = [0.01, 0.1, 1.0, 10.0]

plt.figure()

for lr in learning_rates:

    model = Adaline(learning_rate=lr, max_iterations=300)

    train_mse = model.fit(X_train, y_train)

    plt.plot(np.clip(train_mse, 1e-8, None), label=f"lr={lr}")

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Effect of Learning Rate on Convergence")

plt.legend()
plt.grid(True)

plt.savefig(f"{PLOT_DIR}/learning_rate_experiment.png")
plt.close()

# ----------------------------
# Training size vs Final Loss
# ----------------------------
fractions = np.arange(0.1, 1.1, 0.1)

final_losses = []

for frac in fractions:

    n = int(frac * len(X_train))

    idx = np.random.choice(len(X_train), n, replace=False)

    X_sub = X_train[idx]
    y_sub = y_train[idx]

    model = Adaline(learning_rate=0.01, max_iterations=500)

    model.fit(X_sub, y_sub)

    val_loss = model.score(X_val, y_val)

    final_losses.append(val_loss)

plt.figure()

plt.plot(fractions * 100, final_losses, marker="o")

plt.xlabel("Training Set Size (%)")
plt.ylabel("Validation MSE")
plt.title("Training Set Size vs Final Validation Loss")

plt.grid(True)

plt.savefig(f"{PLOT_DIR}/training_size_vs_loss.png")
plt.close()

print("All ADALINE experiments completed.")
print("Plots saved in:", PLOT_DIR)