import pandas as pd

from source import plot_learning_curve


def main():
    history_file = "histories/120/history_7n_2l_1-2000.csv"

    history_loaded = pd.read_csv(history_file)
    plot_learning_curve(history_loaded)


if __name__ == "__main__":
    main()
