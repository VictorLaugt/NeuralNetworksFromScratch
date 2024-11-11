from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from neural_network import NeuralNetwork

from flat_neural_network import FlatNeuralNetwork
from deep_neural_network import DeepNeuralNetwork

from activation_functions import Sigmoid, LeakyRelu
from loss_functions import SquaredError

import dataset.mnist as data
from metrics import compute_accuracy, plot_confusion_matrix

import matplotlib.pyplot as plt
from time import perf_counter


def compare_neural_networks(neural_networks: Sequence[NeuralNetwork], names: Sequence[str]) -> None:
    assert len(neural_networks) == len(names)

    fig, ax = plt.subplots(1, len(neural_networks)+1, figsize=(20, 5))
    for i, (nn, name) in enumerate(zip(neural_networks, names), start=1):
        print(f"Training {name}:")

        # train the neural network
        t0 = perf_counter()
        loss_values = nn.fit(data.X_train, data.d_train, criterion, n_epochs=100, batch_size=512, lr=0.01)
        ax[0].plot(loss_values, label=name)
        t1 = perf_counter()

        # evaluate the trained neural network
        labels_train_pred = nn.predict(data.X_train)
        labels_test_pred = nn.predict(data.X_test)

        accuracy_train = compute_accuracy(data.labels_train, labels_train_pred)
        accuracy_test = compute_accuracy(data.labels_test, labels_test_pred)
        print(f"train accuracy = {accuracy_train}")
        print(f"test accuracy  = {accuracy_test}")
        print(f"number of parameters = {nn.n_parameters()}")
        print(f"training time = {t1 - t0:.5f}\n")

        ax[i].set_title(name)
        plot_confusion_matrix(data.labels_test, labels_test_pred, ax=ax[i])

    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("training loss")
    ax[0].grid()
    ax[0].legend()

    fig.tight_layout()
    plt.show()


g = Sigmoid()
flat_nn = FlatNeuralNetwork(
    n_input=data.height*data.width,
    n_hidden=32,
    n_output=10,
    activation1=g,
    activation2=g
)
deep_nn = DeepNeuralNetwork(
    layer_sizes=(data.height*data.width, 32, 64, 32, 10),
    activations=(g, g, g, g),
)
criterion = SquaredError()

compare_neural_networks((flat_nn, deep_nn), ("flat neural network", "deep neural network"))
