import numpy as np

import csv
from urllib.request import urlopen

n_visible = 15 #43
n_hidden = 6

rng = np.random.default_rng(42)

W = rng.normal(0, 0.01, size=(n_visible, n_hidden))
b = np.zeros(n_visible)
c = np.zeros(n_hidden)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_hidden(v, W, c):
    p_h = sigmoid(v @ W + c)
    h = (np.random.rand(len(p_h)) < p_h).astype(int)
    return h, p_h


def sample_visible(h, W, b):
    p_v = sigmoid(W @ h + b)
    v = (np.random.rand(len(p_v)) < p_v).astype(int)
    return v, p_v


# def predict_class(v_features, W, b, c, steps=100):
#     v = np.zeros(15)
#     v[:12] = v_features  # הצמדת התכונות
#
#     for _ in range(steps):
#         h, _ = sample_hidden(v, W, c)
#         v_new, _ = sample_visible(h, W, b)
#         v[12:] = v_new[12:]  # רק נוירוני הזן חופשיים
#
#     return np.argmax(v[12:])

def deduce_class(v_features, W, b, c, steps=200):
    """
    v_features: length-12 binary vector
    returns: predicted class index (0,1,2)
    """

    v = np.zeros(15, dtype=int)
    v[:12] = v_features  # clamp features

    class_counts = np.zeros(3)

    for _ in range(steps):
        h, _ = sample_hidden(v, W, c)
        v_new, _ = sample_visible(h, W, b)

        # re-clamp feature neurons
        v[:12] = v_features

        # update only class neurons
        v[12:] = v_new[12:]

        class_counts += v[12:]

    return np.argmax(class_counts)



def train_rbm(data, W, b, c, lr=0.05, epochs=200):
    for epoch in range(epochs):
        for v0 in data:
            # Positive phase
            h0, ph0 = sample_hidden(v0, W, c)

            # Negative phase
            v1, pv1 = sample_visible(h0, W, b)
            h1, ph1 = sample_hidden(v1, W, c)

            # Updates
            W += lr * (np.outer(v0, ph0) - np.outer(v1, ph1))
            b += lr * (v0 - v1)
            c += lr * (ph0 - ph1)


IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


def load_iris():
    data = []
    labels = []

    with urlopen(IRIS_URL) as f:
        reader = csv.reader(line.decode("utf-8") for line in f)
        for row in reader:
            if len(row) != 5:
                continue
            features = list(map(float, row[:4]))
            label = row[4]
            data.append(features)
            labels.append(label)

    return np.array(data), labels


def compute_bins(data, n_bins=3):
    """
    data: shape (N, 4)
    returns: list of 4 arrays, each containing bin edges
    """
    bins = []
    for feature_idx in range(data.shape[1]):
        feature = data[:, feature_idx]
        edges = np.quantile(feature, np.linspace(0, 1, n_bins + 1))
        bins.append(edges)
    return bins


def discretize_one_hot(data, bins):
    """
    data: (N, 4)
    bins: output of compute_bins
    returns: (N, 12) binary matrix
    """
    N = data.shape[0]
    one_hot = np.zeros((N, 12), dtype=int)

    for i in range(N):
        for f in range(4):
            value = data[i, f]
            edges = bins[f]

            # find bin index
            bin_idx = np.searchsorted(edges, value, side="right") - 1
            bin_idx = min(bin_idx, 2)  # safety clamp

            neuron_idx = f * 3 + bin_idx
            one_hot[i, neuron_idx] = 1

    return one_hot


def encode_labels(labels):
    label_map = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    Y = np.zeros((len(labels), 3), dtype=int)
    for i, lbl in enumerate(labels):
        Y[i, label_map[lbl]] = 1

    return Y

def print_accuracy(visible_data):
    correct = 0
    total = visible_data.shape[0]

    for idx in range(total):
        sample = visible_data[idx]

        true_class = np.argmax(sample[12:])

        predicted_class = deduce_class(
            v_features=sample[:12],
            W=W,
            b=b,
            c=c,
            steps=300
        )

        if predicted_class == true_class:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.3f}")


def main():
    # Load data
    X_raw, labels = load_iris()

    # Discretize
    bins = compute_bins(X_raw, n_bins=3)
    X_features = discretize_one_hot(X_raw, bins)   # (150, 40)
    Y_classes = encode_labels(labels)               # (150, 3)

    # Full visible layer
    visible_data = np.hstack([X_features, Y_classes])  # (150, 43)

    # print("Visible shape:", visible_data.shape)
    # print("Example sample:", visible_data[0])

    # visible_data shape: (150, 15)
    # idx = np.random.randint(0, visible_data.shape[0])
    #
    # sample = visible_data[idx]
    #
    # true_class = np.argmax(sample[12:])
    #
    # predicted_class = deduce_class(
    #     v_features=sample[:12],
    #     W=W,
    #     b=b,
    #     c=c,
    #     steps=300
    # )
    #
    # print("Sample index:", idx)
    # print("True class:", true_class)
    # print("Predicted class:", predicted_class)

    print_accuracy(visible_data)

    train_rbm(
        data=visible_data,
        W=W,
        b=b,
        c=c,
        lr=0.05,
        epochs=1000
    )

    print_accuracy(visible_data)


if __name__ == "__main__":
    main()

    # TODO: Replace all the 3 and 12 with 10 and 40 for more granular classification
    # ALso replace the 2 with a 9, meant to access the highest index we can before overlapping

