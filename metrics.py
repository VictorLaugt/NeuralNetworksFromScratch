import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, class_names=None, ax=None):
    labels = np.unique((y_true, y_pred))
    n_labels = labels.max() + 1

    conf_mat = np.zeros((n_labels, n_labels), dtype=int)
    for true, pred in zip(y_true, y_pred):
        conf_mat[true, pred] += 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.matshow(conf_mat, cmap='Blues')
    ax.set_xticks(np.arange(conf_mat.shape[1]))
    ax.set_yticks(np.arange(conf_mat.shape[0]))

    if class_names is not None:
        assert len(class_names) == n_labels, 'wrong number of labels'
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, conf_mat[i, j], ha='center', va='center', color='white' if conf_mat[i, j] > conf_mat.max() / 2 else 'black')

    return ax


def compute_accuracy(y_true, y_pred):
    assert len(y_pred) == len(y_true) and y_pred.ndim == y_true.ndim
    return np.count_nonzero(y_pred == y_true) / len(y_true)


if __name__ == '__main__':
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(y_true, y_pred, class_names=['Class 0', 'Class 1'], ax=ax)
    print(f"accuracy = {compute_accuracy(y_true, y_pred)}")
    plt.show()
