# Introduction
This is my masters disseratation project. The object of this project is to use LSTM model to predict MRT demands in Singapore. The dataset used in the project is the real MRT dataset. MapReduce is used to process the raw data. Keras and TensorFlow is used to build the prediction model. **This is an on-going project.**

# Files
- **mrt_data_processing/src/main/java/:**
	- **PassengerCount.java:** the main MapReduce programme to count no. of passengers in different time windows
	- **Util.java:** utility functions
- **source/:**
	- **run_training.py:** Train a LSTM model for a single MRT station
	- **train_multiple_stations.py:** Train LSTM model for multiple MRT stations
	- **predict.py:** Load a trained model, make predictions and plot.
	- **plot_history.py:** plot training and testing error history
	- **utils/:** utility functions