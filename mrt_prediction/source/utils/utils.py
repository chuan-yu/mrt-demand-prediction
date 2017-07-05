import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from math import sqrt


def load_raw_data(filename):
    raw_data = pd.read_csv(filename, header=None, index_col=0, squeeze=True)
    return raw_data


# convert a timeseries to supervised leanring data
def timeseries_to_supervised(series):
    shifted_series = series.shift(-1)
    df = pd.concat([series, shifted_series], axis=1)
    df.fillna(0, inplace=True)
    return df


# scale data to (0, 1) range
def scale(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data


def get_scaled_train_test(filename, train_ratio=0.7):
    raw_data = load_raw_data(filename)
    supervised = timeseries_to_supervised(raw_data)
    supervised_value = supervised.values
    scaler, scaled_supervised_values = scale(supervised_value)

    train_no = int(scaled_supervised_values.shape[0] * train_ratio)
    train = scaled_supervised_values[0:train_no]
    test = scaled_supervised_values[train_no:]

    return train, test, scaler


# invert data to its original scale
def invert_scale(scaler, X, y):
    value_pair = [X] + [y]
    array = np.array(value_pair)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def build_lstm_model_1_layer(neurons, batch_size, timestep, features):
    model = Sequential()
    model.add(LSTM(neurons,
                   batch_input_shape=(batch_size, timestep, features),
                   stateful=True, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def build_lstm_model_2_layers(neurons, batch_size, timestep, features):
    model = Sequential()
    model.add(LSTM(neurons,
                   batch_input_shape=(batch_size, timestep, features),
                   stateful=True, return_sequences=True))
    model.add(LSTM(neurons, return_sequences=False, stateful=True))
    model.add(Dense(1))
    optimizer = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer="adam")
    return model


def build_lstm_model_3_layers(neurons, batch_size, timestep, features):
    model = Sequential()
    model.add(LSTM(neurons,
                   batch_input_shape=(batch_size, timestep, features),
                   stateful=True, return_sequences=True))
    model.add(LSTM(neurons, return_sequences=True, stateful=True))
    model.add(LSTM(neurons, return_sequences=False, stateful=True))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def fit_lstm(model, train, test, batch_size, timesteps,
             nb_epoch, checkpoint_folder=None,
             validation=False, checkpoints=False):
    # prepare data
    X, y = train[:, 0:-1], train[:, -1]
    X_test, y_test = test[:, 0:-1], test[:, -1]
    X = X.reshape(X.shape[0], timesteps, X.shape[1])
    X_test = X_test.reshape(X_test.shape[0], timesteps, X_test.shape[1])

    print(X.shape)
    # add checkpoint
    callback_list = None
    if checkpoints:
        filepath = checkpoint_folder + "weights.{loss:.5f}.hdf5"
        if validation:
            filepath = checkpoint_folder + "weights.{val_loss:.5f}.hdf5"

        monitor = 'loss'
        if validation:
            monitor = 'val_loss'
        checkpoint = ModelCheckpoint(filepath, monitor=monitor, verbose=1,
                                     save_best_only=True, mode="min")
        callback_list = [checkpoint]

    # set validation data
    validation_data = None
    if validation:
        validation_data = (X_test, y_test)

    # fit

    val_loss = []
    loss = []
    for i in range(nb_epoch):
        print("epoch %s/%s" % (str(i), str(nb_epoch)))
        history = model.fit(X, y,
                            epochs=1, batch_size=batch_size,
                            verbose=0, shuffle=False,
                            callbacks=callback_list,
                            validation_data=validation_data)
        val_loss.append(history.history['val_loss'][0])
        loss.append(history.history['loss'][0])
        model.reset_states()

    histories = {"val_loss": val_loss, "loss": loss}

    return histories


def build_up_state(model, train, batch_size):
    train_reshaped = train[:, 0].reshape(len(train), 1, 1)
    model.predict(train_reshaped, batch_size)
    return model


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


def plot_comparison(expectations, predictions):
    plt.plot(expectations, color='blue')
    plt.plot(predictions, color='orange')
    plt.title("expectations vs. predictions")
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
