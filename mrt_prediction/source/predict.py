from datetime import datetime, timedelta
from source.utils.utils import *

def main():
    lr = 0.001
    l2_coef = 0
    dropout = 0
    station_name = 16
    data_file = "../data/hour/%s.csv" % station_name
    time_steps = 16
    model_file = "../checkpoints/hour/raw/%s.h5" % station_name

    # load scaled train and test data
    raw_data = load_raw_data(data_file)

    _, test_X, _, test_y, test_index = prepare_data(raw_data, time_steps, 0.3)

    if model_file:
        # build lstm model
        layers = [128, 256, 512]
        input_shape = (test_X.shape[1], test_X.shape[2])
        model = build_model(layers, input_shape, lr, l2_coef, dropout, batch_normalization=False)
        model.summary()

        # load saved weights
        model.load_weights(model_file)

        # make predictions for test data
        predictions = model.predict(test_X)
        predictions = predictions.reshape(predictions.shape[0])
        predictions = pd.Series(predictions, index=test_index)

        test_y = test_y.reshape(test_y.shape[0])
        expectations = pd.Series(test_y, index=test_index)

        # report error
        rmse = get_rmse(expectations.values, predictions.values)
        print('Test RMSE: %.3F' % rmse)

        # plot predictions and expectations
        _, scaled_raw_data = scale(raw_data[raw_data.columns[0]])
        plot_comparison(scaled_raw_data, predictions, station_name)

    else:
        raise ValueError('empty model file name')

if __name__ == "__main__":
    main()
