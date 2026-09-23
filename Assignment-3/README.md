# Assignment 3: Effect of Learning Rate and Epochs on MLP Performance

## Aim

To implement a Multilayer Perceptron (MLP) model for Iris dataset classification and analyze the effect of different learning rates and numbers of epochs on model performance.

## Dataset

The Iris dataset is used for this assignment.

- Number of samples: 150
- Number of features: 4
- Number of classes: 3
- Classes: Setosa, Versicolor, Virginica

## Technologies and Libraries Used

- Python
- TensorFlow
- Keras
- Scikit-learn
- Pandas

### Libraries

- TensorFlow / Keras – for building and training the MLP model
- Scikit-learn – for dataset loading, train-test splitting, and feature scaling
- Pandas – for storing and displaying experimental results

## Data Preprocessing

The Iris dataset is divided into training and testing sets using an 80:20 ratio.

StandardScaler is used to standardize the input features before training the model.

## MLP Model

The neural network consists of:

- Input Layer – 4 input features
- Hidden Layer 1 – 10 neurons with ReLU activation
- Hidden Layer 2 – 8 neurons with ReLU activation
- Output Layer – 3 neurons with Softmax activation

## Model Compilation

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy

## Experiment

The model is trained using different learning rates and numbers of epochs.

### Learning Rates

- 0.0001
- 0.001
- 0.01

### Epochs

- 10
- 30
- 50

A total of 9 combinations of learning rate and epochs are tested.

The performance of each combination is evaluated using:

- Test Accuracy
- Test Loss

## Results

The results for different learning rates and epochs are stored in a Pandas DataFrame containing:

- Learning Rate
- Epochs
- Test Accuracy
- Test Loss

This allows comparison of how learning rate and training duration affect the performance of the MLP model.

## Conclusion

The experiment demonstrates the effect of learning rate and number of epochs on the performance of an MLP classification model. Different combinations are trained and evaluated using test accuracy and test loss.
