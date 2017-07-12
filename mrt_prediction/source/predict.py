from datetime import datetime, timedelta
from source.utils.utils import *

def main():
    data_file = "../data/120.csv"
    time_steps = 16
    model_file = "../checkpoints/120/(256-256-256-ts16).h5"

    # load scaled train and test data
    raw_data = load_raw_data(data_file)
    _, test_X, _, test_y = prepare_data(raw_data, time_steps, 0.3)

    if model_file:
        # build lstm model
        layers = [256, 256, 256]
        input_shape = (test_X.shape[1], test_X.shape[2])
        model = build_model(layers, input_shape)
        model.summary()

        # load saved weights
        model.load_weights(model_file)

        # make predictions for test data
        predictions = model.predict(test_X)

        d1 = datetime(2016, 3, 22, 16)
        d2 = datetime(2016, 4, 1, 0)
        delta = d2 - d1
        hours = delta.days * 24 + int(delta.seconds / 3600)
        datetime_list = [d1 + timedelta(hours=i) for i in range(hours)]

        predictions = predictions.reshape(predictions.shape[0])
        predictions = pd.Series(predictions)
        predictions.index = datetime_list
        test_y = test_y.reshape(test_y.shape[0])
        expectations = pd.Series(test_y)
        expectations.index = datetime_list

        # report error
        rmse = get_rmse(expectations.values, predictions.values)
        print('Test RMSE: %.3F' % rmse)

        # plot predictions and expectations
        plot_comparison(expectations, predictions)

    else:
        raise ValueError('empty model file name')

if __name__ == "__main__":
    main()
