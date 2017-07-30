from matplotlib import pyplot as plt


def plot_learning_curve(history):
    plt.close()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.axhline(0.0023, color='red')
    plt.title('learning curve')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'baseline'], loc='upper right')
    plt.show()


def plot_comparison(expectations, predictions, station_name):
    plt.plot(expectations, color='blue')
    plt.plot(predictions, color='orange')
    plt.title(station_name)
    plt.ylabel('Scaled_Demand')
    plt.xlabel('Timestep')
    plt.legend(['expectations', 'predictions'], loc='upper right')
    plt.show()

