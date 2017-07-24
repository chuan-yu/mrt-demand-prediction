from utils.utils import *
import os
import csv


def main():
    layers = [128, 256, 512]
    batch_size = 252
    lr = 0.001
    l2_coef = 0
    dropout = 0
    epochs = 1000
    time_steps = 16
    batch_normalization = False
    data_folder = "../data/hour/"
    checkpoint_folder = "../checkpoints/hour/"
    tb_main = "../tf_checkpoints/hour/"
    result_file = "../results/results.csv"
    verbose = 0

    data_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    result_dict = {}

    for file in data_files:
        # load data
        raw_data = load_raw_data(data_folder + file)
        train_X, test_X, train_y, test_y = prepare_data(raw_data, time_steps, 0.3)

        # build lstm model
        input_shape = (train_X.shape[1], train_X.shape[2])
        model = build_model(layers, input_shape, lr,
                            l2_coef, dropout,
                            batch_normalization=batch_normalization)

        # fit the model and get training history
        station_name = file.replace('.csv', '')
        checkpoint_name = "%s.h5" % station_name
        tb_folder = tb_main + "%s-(128, 256, 512)" % station_name
        print("Running training for %s ..." % station_name)
        history = fit_model(model, (train_X, train_y), (test_X, test_y),
                            batch_size, epochs, checkpoint_folder,
                            checkpoint_name, tb_folder, verbose=verbose)

        # load the best weights and make predictions
        best_weights = checkpoint_folder + checkpoint_name
        print("Loading weights %s" % best_weights)
        model.load_weights(best_weights)
        predictions = model.predict(test_X, batch_size=batch_size)

        # get rmse for the station
        rmse = get_rmse(test_y, predictions)
        print("RMSE for %s is %s" % (station_name, str(rmse)))
        result_dict[station_name] = rmse

    # save results to a csv file
    with open(result_file, 'wb') as file:
        writer = csv.writer(file)
        for key, value in result_dict.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    main()
