from source.utils.utils import *


def main():
    data_file = "./data/120_time.csv"
    checkpoint_folder = "./checkpoints/models/"
    tensorboard_folder = "./tf_checkpoints/120-MF-(400, 400, 400)-ts1-l20.001/"
    history_file = "./histories/history.csv"
    layers = [400, 400, 400]
    batch_size = 224
    l2_coef = 0.001
    dropout = 0
    epochs = 1000
    time_steps = 1
    model_file = ""

    # load scaled train and test data
    raw_data = load_raw_data(data_file)
    train_X, test_X, train_y, test_y = prepare_data(raw_data, time_steps, 0.3)

    # build lstm model
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = build_model(layers, input_shape, l2_coef, dropout)
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
