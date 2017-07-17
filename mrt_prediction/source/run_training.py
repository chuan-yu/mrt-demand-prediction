from source.utils.utils import *


def main():
    data_file = "./data/45.csv"
    checkpoint_folder = "./checkpoints/models/"
    tensorboard_folder = "./tf_checkpoints/45-(16, 32, 64)-ts16/"
    history_file = "./histories/history.csv"
    layers = [16, 32, 64]
    batch_size = 252
    lr = 0.001
    l2_coef = 0
    dropout = 0
    epochs = 1000
    time_steps = 16
    model_file = ""

    # load scaled train and test data
    raw_data = load_raw_data(data_file)

    train_X, test_X, train_y, test_y = prepare_data(raw_data, time_steps, 0.3)

    # build lstm model
    input_shape = (train_X.shape[1], train_X.shape[2])
    # batch_input_shape = (batch_size, train_X.shape[1], train_X.shape[2])
    model = build_model(layers, input_shape, lr, l2_coef, dropout, batch_normalization=False)
    model.summary()

    # initialize the model with saved weights
    if model_file:
        model.load_weights(model_file)

    # fit the model and get training history
    history = fit_model(model, (train_X, train_y), (test_X, test_y),
                        batch_size, epochs, checkpoint_folder, tensorboard_folder)

    # make predictions for test data
    predictions = model.predict(test_X, batch_size=batch_size)

    # report error
    rmse = get_rmse(test_y, predictions)
    print('Test RMSE: %.3F' % rmse)

    plot_comparison(test_y, predictions)


if __name__ == "__main__":
    main()
