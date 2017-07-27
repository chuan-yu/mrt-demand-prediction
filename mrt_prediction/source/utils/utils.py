import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from math import sqrt


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


def build_model(layers, input_shape, lr, l2_coef, dropout=0, batch_normalization=False):
    model = Sequential()
    regularizer = l2(l2_coef)
    for i, layer in enumerate(layers):
        return_sequence = True if i < len(layers) - 1 else False
        if i == 0:
            model.add(LSTM(layer, input_shape=input_shape, dropout=dropout,
                           return_sequences=return_sequence,
                           kernel_regularizer=regularizer))
        else:
            model.add(LSTM(layer, dropout=dropout,
                           return_sequences=return_sequence,
                           kernel_regularizer=regularizer))
        if batch_normalization:
            if i < len(layers) - 1:
                model.add(BatchNormalization())

    model.add(Dense(1))
    optimizer = Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def fit_model(model, train, test, batch_size, epochs, checkpoint_dir, checkpoint_name, tb_dir=None, verbose=1):
    train_X, train_y, test_X, test_y = train[0], train[1], test[0], test[1]

    call_back_list = None
    tb_call_back = None
    if tb_dir:
        tb_call_back = TensorBoard(log_dir=tb_dir, histogram_freq=0, write_graph=False)

    checkpoint = None
    if checkpoint_dir:
        checkpoint_file = checkpoint_dir + checkpoint_name
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=verbose,
                                     save_best_only=True, mode="min")

    if tb_call_back or checkpoint:
        call_back_list = []

        if tb_call_back:
            call_back_list.append(tb_call_back)

        if checkpoint:
            call_back_list.append(checkpoint)


    history = model.fit(train_X, train_y, epochs=epochs,
                        batch_size=batch_size, verbose=verbose,
                        validation_data=(test_X, test_y),
                        callbacks=call_back_list)

    history = {'val_loss': history.history['val_loss'],
               'loss': history.history['loss']}
    return history



def plot_learning_curve(history):
    plt.close()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.axhline(0.0023, color='red')
    plt.title('learning curve')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'baseline'], loc='upper right')
    plt.show()


def plot_comparison(expectations, predictions, station_name):
    plt.plot(expectations, color='blue')
    plt.plot(predictions, color='orange')
    plt.title(station_name)
    plt.ylabel('Scaled_Demand')
    plt.xlabel('Timestep')
    plt.legend(['expectations', 'predictions'], loc='upper right')
    plt.show()


def one_step_predict_lstm(model, batch_size, X):
    X = X.reshape(1, 1, 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def predict_lstm(model, batch_size, test):
    predictions = []
    expectations = []
    for i in range(len(test)):
        X, y = test[i, 0], test[i, 1]
        yhat = one_step_predict_lstm(model, batch_size, X)
        predictions.append(yhat)
        expectations.append(y)
    return predictions, expectations


def get_rmse(expectations, predictions):
    rmse = sqrt(mean_squared_error(expectations, predictions))
    return rmse
