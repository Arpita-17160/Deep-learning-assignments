# Assignment 4: LSTM-Based Time-Series Forecasting

## Aim

To develop an LSTM-based model for time-series forecasting using an air passenger dataset.

## Dataset

The Air Passengers dataset is used for this assignment.

The dataset contains monthly passenger numbers over time and is used to predict future passenger values based on previous observations.

## Technologies and Libraries Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Libraries

- Pandas – for loading and handling the dataset
- NumPy – for numerical operations and preparing sequences
- TensorFlow / Keras – for building and training the LSTM model
- Scikit-learn – for Min-Max normalization
- Matplotlib – for visualizing actual and predicted values

## Data Preprocessing

The passenger values are normalized using MinMaxScaler.

A time-step of 12 months is used to create input sequences. The previous 12 months of passenger data are used to predict the passenger value for the next month.

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

## LSTM Model

The model consists of:

- LSTM Layer – 50 units with ReLU activation
- Dense Output Layer – 1 neuron

## Model Compilation

- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

## Model Training

The model is trained using:

- Epochs: 20
- Batch Size: 16
- Time Steps: 12

## Prediction and Visualization

After training, the model predicts passenger values for the test dataset.

The predicted values are converted back to their original scale and compared with the actual passenger values.

A graph is plotted to visualize the difference between:

- Actual Passenger Values
- Predicted Passenger Values

## Result

The LSTM model is used to forecast future passenger numbers based on the previous 12 months of observations.

The actual and predicted values are visualized using a time-series plot.

## Conclusion

The experiment demonstrates how an LSTM neural network can be used for time-series forecasting. The model learns patterns from previous passenger observations and uses them to predict future values.
