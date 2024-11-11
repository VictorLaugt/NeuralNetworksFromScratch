import numpy as np
from neural.structures.neural_network import NeuralNetwork


class FlatNeuralNetwork(NeuralNetwork):
    def __init__(self, n_input, n_hidden, n_output, activation1, activation2, rng=None):
        self.n_inputs = n_input
        self.n_hiddens = n_hidden
        self.n_outputs = n_output
        self.activ1 = activation1
        self.activ2 = activation2
        self.init_parameters(np.random.default_rng(rng))

    def init_parameters(self, rng):
        self.W1, self.b1 = self.activ1.layer_init(rng, self.n_inputs, self.n_hiddens)
        self.W2, self.b2 = self.activ2.layer_init(rng, self.n_hiddens, self.n_outputs)

    def n_parameters(self):
        return self.n_hiddens * (self.n_inputs+1) + self.n_outputs * (self.n_hiddens+1)

    def predict(self, X):
        assert X.ndim == 2, 'wrong rank'
        assert X.shape[0] == self.n_inputs, 'wrong number of inputs'

        y = self._forward(X)
        return np.argmax(y, axis=0)

    def _forward(self, X):
        self.a1 = self.W1 @ X + self.b1
        self.h1 = self.activ1.func(self.a1)

        self.a2 = self.W2 @ self.h1 + self.b2
        y = self.activ2.func(self.a2)
        return y

    def _backward(self, X, y, d, criterion):
        delta2 = self.activ2.deriv(self.a2) * criterion.deriv(y, d)
        self.d_W2 = delta2 @ self.h1.T
        self.d_b2 = np.sum(delta2, axis=1, keepdims=True)

        delta1 = self.activ1.deriv(self.a1) * (self.W2.T @ delta2)
        self.d_W1 = delta1 @ X.T
        self.d_b1 = np.sum(delta1, axis=1, keepdims=True)

    def train_iteration(self, X, d, criterion, lr):
        y = self._forward(X)
        loss = criterion.func(y, d)
        self._backward(X, y, d, criterion)

        self.W1 -= lr * self.d_W1
        self.W2 -= lr * self.d_W2
        self.b1 -= lr * self.d_b1
        self.b2 -= lr * self.d_b2

        return loss

    def fit(self, X, d, criterion, n_epochs, batch_size, lr=0.01):
        assert X.ndim == d.ndim == 2, 'wrong rank'
        assert X.shape[0] == self.n_inputs, 'wrong number of inputs'
        assert d.shape[0] == self.n_outputs, 'wrong number of outputs'
        assert X.shape[1] == d.shape[1], 'wrong number of samples'

        return super().fit(X, d, criterion, n_epochs, batch_size, lr)
