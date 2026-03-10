import numpy as np
from activations import get_activation
from optimizers import SGD

class MLP:
    def __init__(
        self,
        layer_sizes,
        activations,
        learning_rate=0.01,
        optimizer="sgd"
    ):
        """
        layer_sizes: [input, hidden1, ..., output]
        activations: list of activation names per layer (excluding input)
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.L = len(layer_sizes) - 1

        # parameters
        self.W = {}
        self.b = {}

        # caches
        self.Z = {}
        self.A = {}

        # initialize weights (Xavier)
        for l in range(1, self.L + 1):
            fan_in = layer_sizes[l - 1]
            fan_out = layer_sizes[l]
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.W[l] = np.random.uniform(-limit, limit, (fan_out, fan_in))
            self.b[l] = np.zeros((fan_out, 1))

        # activations
        self.act = {}
        self.act_deriv = {}
        for l, name in enumerate(activations, start=1):
            f, df = get_activation(name)
            self.act[l] = f
            self.act_deriv[l] = df

        # optimizer
        self.optimizer = SGD(learning_rate)

    # ------------------
    # Forward pass
    # ------------------
    def forward(self, X):
        self.A[0] = X.T

        for l in range(1, self.L + 1):
            self.Z[l] = self.W[l] @ self.A[l - 1] + self.b[l]
            self.A[l] = self.act[l](self.Z[l])

        return self.A[self.L]

    # ------------------
    # Backward pass
    # ------------------
    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(1, -1)

        grads_W = {}
        grads_b = {}

        dA = 2 * (self.A[self.L] - y)

        for l in reversed(range(1, self.L + 1)):
            dZ = dA * self.act_deriv[l](self.Z[l])
            grads_W[l] = (1 / m) * dZ @ self.A[l - 1].T
            grads_b[l] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = self.W[l].T @ dZ

        return grads_W, grads_b

    # ------------------
    # Training
    # ------------------
    def fit(self, X, y, epochs=100):
        loss_history = []

        for _ in range(epochs):
            # forward
            y_pred = self.forward(X)

            # loss
            loss = np.mean((y_pred - y.reshape(1, -1)) ** 2)
            loss_history.append(loss)

            # backward
            grads_W, grads_b = self.backward(X, y)

            # update
            for l in range(1, self.L + 1):
                self.W[l] -= self.optimizer.lr * grads_W[l]
                self.b[l] -= self.optimizer.lr * grads_b[l]

        return loss_history

    # ------------------
    # Prediction
    # ------------------
    def predict(self, X):
        return self.forward(X).T