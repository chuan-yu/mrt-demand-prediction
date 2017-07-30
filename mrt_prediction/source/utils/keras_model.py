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