import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# load data from csv file
def load_raw_data(filename):
    raw_data = pd.read_csv(filename, index_col=0, header=None)
    raw_data.index = pd.to_datetime(raw_data.index)
    return raw_data


# generate sine wave
def generate_sine_data():
    x = np.linspace(0, 100, 1000)
    sine = np.sin(x)
    sine = sine.reshape(x.shape[0], 1)
    return sine


# Scale data to (0, 1) range
def scale(data):
    index = data.index
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, index=index)
    return scaler, data


# Split the time series data to train and test data
def split_data(data, time_steps, test_ratio=0.3):
    n_train = int(len(data) * (1 - test_ratio))
    train = data[0:n_train]
    test = data[n_train:]
    test_index = test.index
    train = np.array(train)
    test = np.vstack((train[-time_steps:], test))
    return train, test, test_index


# Generate time window inputs and labels
# Example:
#   input: [1 2 3 4 5 6 7], time_steps = 3
#   output: if not label: [[1 2 3], [2 3 4], [3 4 5], [4 5 6]]
#           if label: [[4], [5], [6], [7]]
def generate_rnn_data(data, time_steps, labels=False):
    rnn_data = []
    if labels:
        for i in range(len(data) - time_steps):
            rnn_data.append(data[i + time_steps, 0])
    else:
        for i in range(len(data) - time_steps):
            rnn_data.append(data[i:i + time_steps])

    rnn_data = np.array(rnn_data)
    return rnn_data


# Generate data ready for RNN training and testing
def prepare_data(raw_data, time_steps, test_ratio):
    _, raw_data = scale(raw_data)
    train, test, test_index = split_data(raw_data, time_steps, test_ratio=test_ratio)
    train_X = generate_rnn_data(train, time_steps, labels=False)
    test_X = generate_rnn_data(test, time_steps, labels=False)
    train_y = generate_rnn_data(train, time_steps, labels=True)
    test_y = generate_rnn_data(test, time_steps, labels=True)
    return train_X, test_X, train_y, test_y, test_index
