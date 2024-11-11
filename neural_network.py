import abc
import numpy as np


class NeuralNetwork(abc.ABC):
    @abc.abstractmethod
    def init_parameters(self):
        """Randomly initialize the parameters of the neural network."""

    @abc.abstractmethod
    def n_parameters(self):
        """Returns the number of learnable parameters of the neural network."""

    @abc.abstractmethod
    def predict(self, X):
        """Return the prediction of the neural network."""

    @abc.abstractmethod
    def train_iteration(self, X, d, criterion, learning_rate):
        """Update the parameters of the neural network."""

    def fit(self, X, d, criterion, n_epochs, batch_size, learning_rate):
        """Fit the neural network using the Stochastic Gradient Descent algorithm."""
        n_sample = X.shape[1]
        loss_values = []

        for e in range(n_epochs):
            shuffled_idx = np.random.permutation(np.arange(n_sample))
            X, d = X[:, shuffled_idx], d[:, shuffled_idx]

            epoch_loss = 0.

            for b in range(0, n_sample, batch_size):
                X_batch, d_batch = X[:, b:b+batch_size], d[:, b:b+batch_size]
                epoch_loss += self.train_iteration(X_batch, d_batch, criterion, learning_rate)

            epoch_loss /= n_sample
            loss_values.append(epoch_loss)
            print(f"epoch {e}: loss = {epoch_loss:.5f}")

        return loss_values
