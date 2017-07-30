from utils.data_processing import *
from utils.keras_model import *
from utils.figures import *


def main():
    data_file = "../data/quarter/45.csv"
    checkpoint_folder = "../checkpoints/models/quarter/"
    checkpoint_name = 'model-45-2.h5'
    tensorboard_folder = None #"../tf_checkpoints/45-(32, 32, 32)-ts16-2"
    history_file = "../histories/history.csv"
    layers = [32, 32, 32]
    num_layers = 3
    state_size = 4
    batch_size = 1200
    lr = 0.001
    l2_coef = 0
    dropout = 0
    epochs = 500
    time_steps = 16
    model_file = ""
    verbose = 1

    # load scaled train and test data
    raw_data = load_raw_data(data_file)
    train_X, test_X, train_y, test_y, _ = prepare_data(raw_data, time_steps, 0.3)


    # build lstm model
    input_shape = (train_X.shape[1], train_X.shape[2])
    model = build_model(layers, input_shape, lr, l2_coef, dropout, batch_normalization=False)
    model.summary()

    # initialize the model with saved weights
    if model_file:
        model.load_weights(model_file)

    # fit the model and get training history
    history = fit_model(model, (train_X, train_y), (test_X, test_y),
                        batch_size, epochs, checkpoint_folder,
                        checkpoint_name, tensorboard_folder, verbose=verbose)

    # make predictions for test data
    predictions = model.predict(test_X, batch_size=batch_size)

    # report error
    rmse = get_rmse(test_y, predictions)
    print('Test RMSE: %.3F' % rmse)

    plot_comparison(test_y, predictions)


if __name__ == "__main__":
    main()
