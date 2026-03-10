import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.001, max_iterations=1000):
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.w = None
        self.b = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        train_mse = []
        val_mse = []

        for _ in range(self.max_iter):
            # -------- Predictions --------
            y_train_pred = self.predict(X_train)
            error = y_train_pred - y_train

            # -------- Gradient descent --------
            dw = (2 / n_samples) * X_train.T @ error
            db = (2 / n_samples) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            # -------- Metrics --------
            train_mse.append(np.mean(error ** 2))

            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_mse.append(np.mean((y_val_pred - y_val) ** 2))

        if X_val is not None and y_val is not None:
            return train_mse, val_mse

        return train_mse

    def predict(self, X):
        return X @ self.w + self.b

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)