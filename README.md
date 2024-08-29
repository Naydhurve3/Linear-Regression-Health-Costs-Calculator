# Health Insurance Cost Prediction
This project aims to predict health insurance costs based on various factors such as age, sex, BMI, children, smoking habits, and region. The prediction model is built using TensorFlow and Keras, and is trained to minimize the mean absolute error (MAE) between predicted and actual costs.

## Steps:
## Importing Libraries:

Essential libraries such as numpy, pandas, matplotlib, and tensorflow are imported.
TensorFlow's keras API is used to build the deep learning model.

## Data Acquisition:
The dataset is downloaded from a given URL and loaded into a Pandas DataFrame for analysis.

## Exploratory Data Analysis (EDA):
Descriptive statistics are computed to understand the distribution of features.
Bar plots and scatter plots are used to visualize the relationship between categorical features and the target variable (expenses), as well as between numerical features and expenses.

# Data Preprocessing:
Categorical features are encoded as numerical values using the astype('category') method.
A dictionary is created to keep track of the encoding for future reference.

## Splitting the Data:
The dataset is split into training (80%) and testing (20%) datasets to ensure that the model can generalize to unseen data.
The target variable expenses is separated from the features.

## Normalization:
A normalization layer is built using TensorFlow's layers.Normalization to standardize the input features, improving the model's performance.

## Model Architecture:
A sequential model is built with two hidden layers, each containing 64 neurons with ReLU activation, followed by an output layer with a single neuron.
The model is compiled using the Adam optimizer and the mean absolute error (MAE) loss function, with MAE and mean squared error (MSE) as metrics.

## Model Training:
The model is trained on the training data for 600 epochs with a validation split of 20%. The training process includes monitoring the loss and validation loss.

#3 Model Evaluation:
The model is evaluated on the test dataset to assess its performance. The mean absolute error (MAE) is used as the primary metric.
The model's performance is checked to ensure that the MAE is below 3500, which is the threshold for passing the challenge.

## Predictions and Visualization:
The model's predictions are plotted against the true values to visualize the accuracy of the predictions.
A scatter plot with a reference line is created to show the correlation between predicted and actual insurance costs.

## How to Run
To run this project, simply execute the provided Python script in a Jupyter notebook or any Python environment with TensorFlow installed. Ensure that all required libraries are installed before running the script.

## Conclusion
This project successfully demonstrates the use of deep learning techniques for predicting health insurance costs based on various features. The model achieves an MAE below the required threshold, indicating good performance in predicting insurance expenses.
