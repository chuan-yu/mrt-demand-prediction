from datetime import datetime, timedelta

from source.utils.utils import *


def main():
    raw_data_file_name = "data/45.csv"
    neurons = 7
    batch_size = 1
    timestep = 1
    features = 1
    model_file = "checkpoints/45/weights.0.00143.hdf5"

    # load scaled train and test data
    train, test, scaler = get_scaled_train_test(raw_data_file_name)

    if model_file:
        # build lstm model
        #lstm_model = build_lstm_model_1_layer(neurons, batch_size, timestep, features)
        lstm_model = build_lstm_model_2_layers(neurons, batch_size, timestep, features)
        #lstm_model = build_lstm_model_3_layers(neurons, batch_size, timestep, features)

        # load saved weights
        lstm_model.load_weights(model_file)

        # make predictions for all train data to build up lstm states
        lstm_model = build_up_state(lstm_model, train, batch_size=1)

        # make predictions for test data
        predictions, expectations = predict_lstm(lstm_model, 1, test)

        d1 = datetime(2016, 3, 22, 16)
        d2 = datetime(2016, 4, 1, 0)
        delta = d2 - d1
        hours = delta.days * 24 + int(delta.seconds / 3600)
        datetime_list = [d1 + timedelta(hours=i) for i in range(hours)]

        predictions = pd.Series(predictions)
        predictions.index = datetime_list
        expectations = pd.Series(expectations)
        expectations.index = datetime_list

        # predictions = scaler.inverse_transform(predictions)
        # expectations = scaler.inverse_transform(expectations)

        # plot predictions and expectations
        plot_comparison(expectations, predictions)

        # report error
        rmse = get_rmse(expectations.values, predictions.values)
        print('Test RMSE: %.3F' % rmse)

    else:
        raise ValueError('empty model file name')

if __name__ == "__main__":
    main()
