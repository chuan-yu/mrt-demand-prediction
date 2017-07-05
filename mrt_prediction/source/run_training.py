from source.utils.utils import *


def main():
    raw_data_file_name = "data/120.csv"
    checkpoint_folder = "checkpoints/models/"
    history_file = "histories/history_7n_2l_1-500.csv"
    neurons = 7
    batch_size = 1
    epochs = 500
    timestep = 240
    features = 1
    model_file = ""

    # load scaled train and test data
    train, test, scaler = get_scaled_train_test(raw_data_file_name)
    # build lstm model
    #lstm_model = build_lstm_model_1_layer(neurons, batch_size, timestep, features)
    lstm_model = build_lstm_model_2_layers(neurons, batch_size, timestep, features)
    #lstm_model = build_lstm_model_3_layers(neurons, batch_size, timestep, features)

    # initialize the model with saved weights
    if model_file:
        lstm_model.load_weights(model_file)
        # make predictions for all train data to build up lstm states
        lstm_model = build_up_state(lstm_model, train, batch_size=1)

    # fit the model and get training history
    histories = fit_lstm(lstm_model, train, test, batch_size, timestep, epochs, checkpoint_folder, validation=True, checkpoints=True)
    pd.DataFrame(histories).to_csv(history_file)

    # make predictions for test data
    lstm_model = build_up_state(lstm_model, train, batch_size=1)
    predictions, expectations = predict_lstm(lstm_model, 1, test)

    # report error
    rmse = get_rmse(expectations, predictions)
    print('Test RMSE: %.3F' % rmse)

    # plot learning curve
    history_loaded = pd.read_csv(history_file)
    plot_learning_curve(history_loaded)


if __name__ == "__main__":
    main()
