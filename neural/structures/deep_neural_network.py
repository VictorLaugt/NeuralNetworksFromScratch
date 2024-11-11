import numpy as np
from neural.structures.neural_network import NeuralNetwork


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, layer_sizes, activations, rng=None):
        assert len(layer_sizes) >= 3, 'need at least one hidden layer'
        assert len(activations) == len(layer_sizes) - 1, 'wrong number of activation functions'

        self.depth = len(layer_sizes) - 2  # number of hidden layers

        self.layer_sizes = layer_sizes
        self.activ = activations

        self.a = [None] * (self.depth+1)
        self.h = [None] * (self.depth+1)

        self.W = [None] * (self.depth+1)
        self.b = [None] * (self.depth+1)

        self.d_W = [None] * (self.depth+1)
        self.d_b = [None] * (self.depth+1)

        self.init_parameters(np.random.default_rng(rng))

    def init_parameters(self, rng):
        for l in range(0, self.depth+1, 1):
            W, b = self.activ[l].layer_init(rng, self.layer_sizes[l], self.layer_sizes[l+1])
            self.W[l] = W
            self.b[l] = b

    def n_parameters(self):
        n_params = 0
        for l in range(0, self.depth+1, 1):
            n_params += self.layer_sizes[l+1] * (self.layer_sizes[l]+1)
        return n_params

    def predict(self, X):
        assert X.ndim == 2, 'wrong rank'
        assert X.shape[0] == self.W[0].shape[1], 'wrong number of inputs'

        y = self._forward(X)
        return np.argmax(y, axis=0)

    def _forward(self, X):
        self.a[0] = self.W[0] @ X + self.b[0]
        self.h[0] = self.activ[0].func(self.a[0])

        for l in range(0, self.depth, 1):
            self.a[l+1] = self.W[l+1] @ self.h[l] + self.b[l+1]
            self.h[l+1] = self.activ[l+1].func(self.a[l+1])

        return self.h[self.depth]

    def _backward(self, X, y, d, criterion):
        delta = self.activ[self.depth].deriv(self.a[self.depth]) * criterion.deriv(y, d)

        for l in range(self.depth-1, -1, -1):
            self.d_W[l+1] = delta @ self.h[l].T
            self.d_b[l+1] = np.sum(delta, axis=1, keepdims=True)
            delta = self.activ[l].deriv(self.a[l]) * (self.W[l+1].T @ delta)

        self.d_W[0] = delta @ X.T
        self.d_b[0] = np.sum(delta, axis=1, keepdims=True)

    def train_iteration(self, X, d, criterion, lr):
        y = self._forward(X)
        loss = criterion.func(y, d)
        self._backward(X, y, d, criterion)

        for l in range(0, self.depth+1, 1):
            self.W[l] -= lr * self.d_W[l]
            self.b[l] -= lr * self.d_b[l]

        return loss

    def fit(self, X, d, criterion, n_epochs, batch_size, lr=0.01):
        assert X.ndim == d.ndim == 2, 'wrong rank'
        assert X.shape[0] == self.W[0].shape[1], 'wrong number of inputs'
        assert d.shape[0] == self.W[self.depth].shape[0], 'wrong number of outputs'
        assert X.shape[1] == d.shape[1], 'wrong number of samples'

        return super().fit(X, d, criterion, n_epochs, batch_size, lr)
