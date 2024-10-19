import numpy as np
from neural_network import NeuralNetwork


class FlatNeuralNetwork(NeuralNetwork):
    def __init__(self, n_input, n_hidden, n_output, activation1, activation2):
        self.n_inputs = n_input
        self.n_hiddens = n_hidden
        self.n_outputs = n_output
        self.activ1 = activation1
        self.activ2 = activation2
        self.init_weights()

    def init_weights(self):
        self.W1 = np.random.randn(self.n_hiddens, self.n_inputs)
        self.b1 = np.random.randn(self.n_hiddens, 1)
        self.W2 = np.random.randn(self.n_outputs, self.n_hiddens)
        self.b2 = np.random.randn(self.n_outputs, 1)

        # self.W1 = np.random.random((self.n_hidden, self.n_input))
        # self.b1 = np.random.random((self.n_hidden, 1))
        # self.W2 = np.random.random((self.n_output, self.n_hidden))
        # self.b2 = np.random.random((self.n_output, 1))

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


if __name__ == '__main__':
    from activation_functions import Sigmoid, LeakyRelu
    from loss_functions import SquaredError

    import dataset.mnist as data

    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    nn = FlatNeuralNetwork(
        n_input=data.height*data.width,
        n_hidden=32,
        n_output=10,
        activation1=Sigmoid(),
        activation2=Sigmoid()
    )

    criterion = SquaredError()
    nn.fit(data.X_train, data.d_train, criterion, n_epochs=10, batch_size=512, lr=0.01)

    def compute_accuracy(y_pred, y):
        assert len(y_pred) == len(y) and y_pred.ndim == y.ndim
        return np.count_nonzero(y_pred == y) / len(y)

    labels_train_pred = nn.predict(data.X_train)
    labels_test_pred = nn.predict(data.X_test)
    accuracy_train = compute_accuracy(labels_train_pred, data.labels_train)
    accuracy_test = compute_accuracy(labels_test_pred, data.labels_test)

    print(f"{accuracy_train = }")
    print(f"{accuracy_test = }")

    disp = ConfusionMatrixDisplay.from_predictions(data.labels_test, labels_test_pred)
    disp.ax_.set_title("One hidden layer neural network classifier")
    plt.show()
