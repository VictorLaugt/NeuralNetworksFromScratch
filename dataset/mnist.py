from pathlib import Path
import numpy as np

MNIST_DATA_DIRECTORY = Path(__file__).resolve().parent

train_images_path = MNIST_DATA_DIRECTORY.joinpath('./mnist-train-images.npy')
train_labels_path = MNIST_DATA_DIRECTORY.joinpath('./mnist-train-labels.npy')
test_images_path = MNIST_DATA_DIRECTORY.joinpath('./mnist-test-images.npy')
test_labels_path = MNIST_DATA_DIRECTORY.joinpath('./mnist-test-labels.npy')

assert train_images_path.is_file(), 'missing data file'
assert train_labels_path.is_file(), 'missing data file'
assert test_images_path.is_file(), 'missing data file'
assert test_labels_path.is_file(), 'missing data file'

images_train = np.load(MNIST_DATA_DIRECTORY.joinpath('./mnist-train-images.npy'))
labels_train = np.load(MNIST_DATA_DIRECTORY.joinpath('./mnist-train-labels.npy'))
images_test = np.load(MNIST_DATA_DIRECTORY.joinpath('./mnist-test-images.npy'))
labels_test = np.load(MNIST_DATA_DIRECTORY.joinpath('./mnist-test-labels.npy'))

height, width = images_train.shape[1:]

n_sample_train = images_train.shape[0]
X_train = images_train.reshape(n_sample_train, height*width).T / 255
d_train = np.eye(10, dtype=labels_train.dtype)[labels_train].T

n_sample_test = images_test.shape[0]
X_test = images_test.reshape(n_sample_test, height*width).T / 255
d_test = np.eye(10, dtype=labels_test.dtype)[labels_test].T


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random

    for i in random.sample(range(n_sample_train), 5):
        plt.title(f"sample {i}, label {labels_train[i]}")
        plt.imshow(X_train[:, i].reshape(height, width), cmap='gray')
        plt.colorbar()
        plt.show()
