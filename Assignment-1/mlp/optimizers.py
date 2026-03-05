import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def init_state(self, model):
        pass  # no state needed

    def step(self, model, grads_W, grads_b):
        updates = {}
        for l in grads_W:
            update_W = self.lr * grads_W[l]
            update_b = self.lr * grads_b[l]
            model.W[l] -= update_W
            model.b[l] -= update_b
            updates[l] = np.mean(np.abs(update_W))
        return updates


class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def init_state(self, model):
        for l in model.W:
            self.vW[l] = np.zeros_like(model.W[l])
            self.vb[l] = np.zeros_like(model.b[l])

    def step(self, model, grads_W, grads_b):
        updates = {}
        for l in grads_W:
            self.vW[l] = self.beta * self.vW[l] + self.lr * grads_W[l]
            self.vb[l] = self.beta * self.vb[l] + self.lr * grads_b[l]
            model.W[l] -= self.vW[l]
            model.b[l] -= self.vb[l]
            updates[l] = np.mean(np.abs(self.vW[l]))
        return updates


class Nesterov:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def init_state(self, model):
        for l in model.W:
            self.vW[l] = np.zeros_like(model.W[l])
            self.vb[l] = np.zeros_like(model.b[l])

    def step(self, model, grads_W, grads_b):
        updates = {}
        for l in grads_W:
            prev_vW = self.vW[l].copy()
            prev_vb = self.vb[l].copy()

            self.vW[l] = self.beta * self.vW[l] + self.lr * grads_W[l]
            self.vb[l] = self.beta * self.vb[l] + self.lr * grads_b[l]

            update_W = -self.beta * prev_vW + (1 + self.beta) * self.vW[l]
            update_b = -self.beta * prev_vb + (1 + self.beta) * self.vb[l]

            model.W[l] -= update_W
            model.b[l] -= update_b
            updates[l] = np.mean(np.abs(update_W))
        return updates


class AdaGrad:
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.cacheW = {}
        self.cacheb = {}

    def init_state(self, model):
        for l in model.W:
            self.cacheW[l] = np.zeros_like(model.W[l])
            self.cacheb[l] = np.zeros_like(model.b[l])

    def step(self, model, grads_W, grads_b):
        updates = {}
        for l in grads_W:
            self.cacheW[l] += grads_W[l] ** 2
            self.cacheb[l] += grads_b[l] ** 2

            update_W = self.lr * grads_W[l] / (np.sqrt(self.cacheW[l]) + self.eps)
            update_b = self.lr * grads_b[l] / (np.sqrt(self.cacheb[l]) + self.eps)

            model.W[l] -= update_W
            model.b[l] -= update_b
            updates[l] = np.mean(np.abs(update_W))
        return updates


class RMSProp:
    def __init__(self, lr=0.01, decay=0.99, eps=1e-8):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cacheW = {}
        self.cacheb = {}

    def init_state(self, model):
        for l in model.W:
            self.cacheW[l] = np.zeros_like(model.W[l])
            self.cacheb[l] = np.zeros_like(model.b[l])

    def step(self, model, grads_W, grads_b):
        updates = {}
        for l in grads_W:
            self.cacheW[l] = self.decay * self.cacheW[l] + (1 - self.decay) * grads_W[l] ** 2
            self.cacheb[l] = self.decay * self.cacheb[l] + (1 - self.decay) * grads_b[l] ** 2

            update_W = self.lr * grads_W[l] / (np.sqrt(self.cacheW[l]) + self.eps)
            update_b = self.lr * grads_b[l] / (np.sqrt(self.cacheb[l]) + self.eps)

            model.W[l] -= update_W
            model.b[l] -= update_b
            updates[l] = np.mean(np.abs(update_W))
        return updates


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW = {}
        self.vW = {}
        self.mb = {}
        self.vb = {}
        self.t = 0

    def init_state(self, model):
        for l in model.W:
            self.mW[l] = np.zeros_like(model.W[l])
            self.vW[l] = np.zeros_like(model.W[l])
            self.mb[l] = np.zeros_like(model.b[l])
            self.vb[l] = np.zeros_like(model.b[l])

    def step(self, model, grads_W, grads_b):
        self.t += 1
        updates = {}
        for l in grads_W:
            self.mW[l] = self.beta1 * self.mW[l] + (1 - self.beta1) * grads_W[l]
            self.vW[l] = self.beta2 * self.vW[l] + (1 - self.beta2) * grads_W[l] ** 2
            self.mb[l] = self.beta1 * self.mb[l] + (1 - self.beta1) * grads_b[l]
            self.vb[l] = self.beta2 * self.vb[l] + (1 - self.beta2) * grads_b[l] ** 2

            mW_hat = self.mW[l] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[l] / (1 - self.beta2 ** self.t)
            mb_hat = self.mb[l] / (1 - self.beta1 ** self.t)
            vb_hat = self.vb[l] / (1 - self.beta2 ** self.t)

            update_W = self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            update_b = self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

            model.W[l] -= update_W
            model.b[l] -= update_b
            updates[l] = np.mean(np.abs(update_W))
        return updates


class Muon:
    """
    Muon (Momentum + Orthogonalization) optimizer.
    Applies Nesterov momentum then orthogonalizes weight updates via
    Newton-Schulz iterations. Biases use plain Nesterov.
    Reference: Kosson et al., 2024 (https://arxiv.org/abs/2409.20325)
    """
    def __init__(self, lr=0.02, beta=0.95, ns_steps=5):
        self.lr = lr
        self.beta = beta
        self.ns_steps = ns_steps
        self.vW = {}
        self.vb = {}

    def init_state(self, model):
        for l in model.W:
            self.vW[l] = np.zeros_like(model.W[l])
            self.vb[l] = np.zeros_like(model.b[l])

    def _newton_schulz(self, G):
        """Orthogonalize 2D matrix G using quintic Newton-Schulz iterations."""
        norm = np.linalg.norm(G)
        if norm < 1e-8:
            return G
        X = G / norm
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(self.ns_steps):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ A @ X)
        return X

    def step(self, model, grads_W, grads_b):
        updates = {}
        for l in grads_W:
            prev_vW = self.vW[l].copy()
            prev_vb = self.vb[l].copy()

            self.vW[l] = self.beta * self.vW[l] + grads_W[l]
            self.vb[l] = self.beta * self.vb[l] + grads_b[l]

            nesterov_W = -self.beta * prev_vW + (1 + self.beta) * self.vW[l]
            nesterov_b = -self.beta * prev_vb + (1 + self.beta) * self.vb[l]

            if nesterov_W.ndim == 2:
                orth_W = self._newton_schulz(nesterov_W)
                scale = max(1e-8, np.sqrt(np.mean(nesterov_W ** 2))) / \
                        max(1e-8, np.sqrt(np.mean(orth_W ** 2)))
                update_W = self.lr * orth_W * scale
            else:
                update_W = self.lr * nesterov_W

            update_b = self.lr * nesterov_b

            model.W[l] -= update_W
            model.b[l] -= update_b
            updates[l] = np.mean(np.abs(update_W))
        return updates