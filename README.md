# Predicting Customer Attrition with Machine Learning

Customer churn or customer attrition is a tendency of clients or customers to abandon a brand and stop being paying clients of a particular business or organization. The percentage of customers discontinuing using a company’s services or products during a specific period is called a customer churn rate. Several bad experiences (or just one) are enough, and a customer may quit. And if a large chunk of unsatisfied customers churn at a time interval, both material losses and damage to reputation would be enormous.

The objective of this project is to understand and predict customer churn for a bank. Specifically, we will initially perform Exploratory Data Analysis (EDA) to identify and visualise the factors contributing to customer churn. This analysis will later help us build Machine Learning models to predict whether a customer will churn or not.

This problem is a typical classification task, the use of recall (i.e. ratio of true positive prediction made out of total positive cases in the dataset) as performance metrics are more relevant in this case, since correctly classifying elements of the positive class (customers who will churn) is more critical for the bank.

## Instructions

The steps for this activity are divided into the following sections:

* Preprocess the data.

* Create a neural network model to predict customer attrition.

* Train and evaluate the neural network model.

#### Preprocess the Data

1. Read the `Churn_Modelling.csv` file from the Resources folder and create a DataFrame.

2. Review the resulting DataFrame. Check the dataset is clearn, then check the data type associated with each column to identify categorical and non-categorical variables.

3. Create a list of categorical variables with an `object` data type.

4. Encode the dataset's categorical variables using the OneHotEncoder module from scikit-learn.

5. Create a new DataFrame named `attrition_df` that contains the encoded categorical variables and the numerical variables from the original dataset.

6. Use `attrition_df` to create the features (`X`) and target (`y`) sets.

7. Checkin correlation among X of the dataset, visualise the result with pairplot.

8. Create the training and testing sets using the `train_test_split` function from scikit-learn.

9. Use StandardScaler to standardise the numerical features.

#### Train and Evaluate the Neural Network Model

1. Create deep neural network model with two hidden layers.

   * Set a variable `number_input_features` equal to the number of input features.

   * Set a variable `hidden_nodes_layer1` equal to an integer number close to the mean of the number of input features and the number of output neurons:

   * Set a variable `hidden_nodes_layer2` equal to an integer number close to the mean of the number of first layer nodes and the output neurons.

   * Add the model’s layers, and then use the `relu` activation function for each hidden layer.
 
   * This model will predict a binary output. Add an output layer with one neuron and use the `sigmoid` activation function.

2. Display the structure of your model using the `summary` function.

3. Compile the model. Use the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` metric.

4. Train (fit) the neural network model with the training data for 100 epochs.

5. Evaluate the model using the test data to determine its loss and accuracy.

#### Modelling using Logistic Regression with the Original Data

1. Fit a logistic regression model by using the training data

2. Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.

3. Evaluate the model’s performance by calculate accuracy scored, confusion matrix and classification report.

#### Modelling using Logistic Regression with the Resampled Training Data

1. Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points.

2. Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.

3. Evaluate the model’s performance by calculate accuracy scored, confusion matrix and classification report.

#### Modelling using Random Forest

1. Making Predictions using the Decsion Tree Model

2. Model Evaluation

3. Fetch the features' importance from the random forest model and display the top 10 most important features.

#### Modelling using K-Nearest Neighbors

1. Instantiate an k-nearest neighbour classifier instance.

2. Fit the model using the training data.

3. Make predictions using the testing data.

4. Generate the classification report for the test data.

#### Modelling using XGBoost

1. conda install -c conda-forge xgboost

2. Initiate XGBClassifier, with 2 trees in the gradient boosting model, maximum depth of each tree being 2, and step size of 2 at which the algorithm learns, specifies "binary:logistic" as the objective function that the model is trying to optimize.

3. Make predictions using the testing data.

4. Generate the confusion matrix and classification report for the test data.

## References

[Keras Sequential model](https://keras.io/api/models/sequential/)

[Keras Dense module](https://keras.io/api/layers/core_layers/dense/)

[Keras evaluate](https://keras.io/api/models/model_training_apis/)

[SKLearn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

---


