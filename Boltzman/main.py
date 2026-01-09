import numpy as np

import csv
from urllib.request import urlopen

# TODO: Increasing the input layer categories from 3 to ten barely helps, same with making the hidden layer larger
# TODO: A hidden layer of 6 performed better than 4 however

INPUT_NEURON_COUNT = 12
OUTPUT_NEURON_COUNT = 3
VISIBLE_LAYER_SIZE = INPUT_NEURON_COUNT + OUTPUT_NEURON_COUNT
HIDDEN_LAYER_SIZE = 6

TRAINING_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


# Function to use for probabilistic part of neuron activation
def sigmoid(value, temperature):
    return 1.0 / (1.0 + np.exp(-value / temperature))


# Function to calculate energy state for stop condition
def energy(visible, hidden, weights, v_bias, h_bias):
    return (
        -visible @ v_bias
        -hidden @ h_bias
        -visible @ weights @ hidden
    )


# Function to get the activation probabilities of the hidden layer based on weights bias and the visible layer
def sample_hidden(visible, weights, hidden_biases, temperature=1):
    field = visible @ weights + hidden_biases
    probability = sigmoid(field, temperature)
    hidden = (np.random.rand(len(probability)) < probability).astype(int)
    return hidden, probability


# Function to get the activation probabilities of the visible layer based on weights bias and the hidden layer
def sample_visible(hidden, weights, visible_biases, temperature=1):
    field = weights @ hidden + visible_biases
    probability = sigmoid(field, temperature)
    visible = (np.random.rand(len(probability)) < probability).astype(int)
    return visible, probability


# Function returns the output neuron activation based on the input data
# For high values of temperature acts very erratically, so we start fairly low already
def deduce_class(
    v_features,
    weights,
    visible_biases,
    hidden_biases,
    max_iters=100,
    initial_temperature=2.0,
    temperature_decrease=0.5,
    flip_threshold=1
):
    # Start with a blank visible layer
    visible = np.random.randint(0, 2, size=15)

    # Load and freeze the input neurons
    visible[:12] = v_features

    # Initialize the temp and copy of visible layer for flip count
    temperature = initial_temperature
    prev_visible = visible.copy()

    index = 0
    for _ in range(max_iters):
        index += 1
        # print(index)

        # Get the new hidden layer
        hidden, _ = sample_hidden(visible, weights, hidden_biases, temperature)

        # Get the new visible layer
        v_new, _ = sample_visible(hidden, weights, visible_biases, temperature)

        # Freeze input neurons again
        visible[:12] = v_features

        # Update only output neurons
        visible[12:] = v_new[12:]

        # Stop condition: network stabilizes
        flips = np.sum(visible != prev_visible)
        # print(flips)
        if flips < flip_threshold:
            # print('broke')
            break

        # Get a new copy
        prev_visible = visible.copy()

        # Reduce temperature by a constant factor
        temperature *= temperature_decrease

    return np.argmax(visible[12:])


# Function to train network on input data
def train_machine(data, weights, visible_biases, hidden_biases, learning_rate=0.05, learning_iterations=200):
    for iteration in range(learning_iterations):
        for initial_visible in data:
            # Get hidden layer and probabilities
            initial_hidden, initial_hidden_probabilities = sample_hidden(initial_visible, weights, hidden_biases, 1)

            # Get visible layer and re shuffle hidden layer
            updated_visible, _ = sample_visible(initial_hidden, weights, visible_biases, 1)
            _, updated_hidden_probabilities = sample_hidden(updated_visible, weights, hidden_biases, 1)

            # Update all the parameters (learning)
            weights += learning_rate * (np.outer(initial_visible, initial_hidden_probabilities) -
                                        np.outer(updated_visible, updated_hidden_probabilities))
            visible_biases += learning_rate * (initial_visible - updated_visible)
            hidden_biases += learning_rate * (initial_hidden_probabilities - updated_hidden_probabilities)


# Function to load the mnist flower data, returns array of data and their labels
def load_data():
    data = []
    labels = []

    reader = csv.reader(line.decode("utf-8") for line in urlopen(TRAINING_DATA_URL))
    for row in reader:
        # Special condition for useless input rows
        if len(row) != 5:
            continue
        features = list(map(float, row[:4]))
        label = row[4]
        data.append(features)
        labels.append(label)

    return np.array(data), labels


# Seperate all the values into the category appropriate for them
def get_feature_categories(data, cat_count):
    example_count, feature_count = data.shape
    data_categories = np.zeros((example_count, feature_count * cat_count), dtype=int)
    category_basket = []

    # For each feature gather the data from all examples to make the appropriate categories
    for feat_num in range(feature_count):
        feature = data[:, feat_num]

        # Compute baskets for each value
        edges = np.quantile(feature, np.linspace(0, 1, cat_count + 1))
        category_basket.append(edges)

        # Assign all samples for this feature at once
        baskets_indices = np.searchsorted(edges, feature, side="right") - 1
        baskets_indices = np.clip(baskets_indices, 0, cat_count - 1)

        # Create the actual categories
        for i in range(example_count):
            neuron_index = feat_num * cat_count + baskets_indices[i]
            data_categories[i, neuron_index] = 1

    return data_categories


# Return the labels as int
def encode_labels(labels):
    label_map = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    label_output = np.zeros((len(labels), 3), dtype=int)
    for i, lbl in enumerate(labels):
        label_output[i, label_map[lbl]] = 1

    return label_output


# Print the overall accuracy of the machine from the entire dataset
def print_accuracy(visible_data, weights, visible_biases, hidden_biases):
    correct = 0
    total = visible_data.shape[0]

    # Loop over all data points and compare the predicted class to the true label
    for index in range(total):
        sample = visible_data[index]

        true_class = np.argmax(sample[12:])

        predicted_class = deduce_class(
            sample[:12],
            weights,
            visible_biases,
            hidden_biases,
        )

        if predicted_class == true_class:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.3f}")


# Sample random example
def sample_random(visible_data, weights, visible_biases, hidden_biases):
    index = np.random.randint(0, visible_data.shape[0])
    sample = visible_data[index]
    true_class = np.argmax(sample[12:])
    predicted_class = deduce_class(
            sample[:12],
            weights,
            visible_biases,
            hidden_biases,
        )

    print(f"For random example in index {index} predicted: {predicted_class} true class is {true_class}")


# Main function to run the machine - including training
def main():
    # Start with random weights matrix of the dimensions of the visible layer X hidden layer
    weights = np.random.default_rng(24).normal(0, 1, size=(VISIBLE_LAYER_SIZE, HIDDEN_LAYER_SIZE))

    # Start with empty biases
    visible_biases = np.zeros(VISIBLE_LAYER_SIZE)
    hidden_biases = np.zeros(HIDDEN_LAYER_SIZE)

    # Load data
    raw_data, labels = load_data()

    # Transform the data into discreet values and in labels
    feature_data = get_feature_categories(raw_data, 3)
    encoded_labels = encode_labels(labels)

    # Compile all the data together
    visible_data = np.hstack([feature_data, encoded_labels])

    # Pick a random example, run deduction and get the overall accuracy
    sample_random(visible_data, weights, visible_biases, hidden_biases)
    print_accuracy(visible_data, weights, visible_biases, hidden_biases)

    # Run learning algorithm to improve deduction
    train_machine(
        visible_data,
        weights,
        visible_biases,
        hidden_biases,
        0.05,
        1000
    )

    # Pick a random example, run deduction and get the overall accuracy
    sample_random(visible_data, weights, visible_biases, hidden_biases)
    print_accuracy(visible_data, weights, visible_biases, hidden_biases)


if __name__ == "__main__":
    main()
