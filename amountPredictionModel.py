import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
import featureEngineering

#Accessing relevant features from the dataset
data = featureEngineering.finalDF[["Cust_Age", "Education", "Marital_Status", "Income", "Wines", "Fruits", "Meat", "Fish", "Sweet", "Gold"]]
data = data.head(1000)
# print(tabulate(data))
print(data.info())

#Handling NaN values.
data['Income'] = data['Income'].fillna(data['Income'].median())
data['Income'] = data['Income'].astype(np.int64)
print(data.info())

#Selecting features and target variable
X = data[["Cust_Age", "Education", "Marital_Status", "Income"]]
y = data[["Wines", "Fruits", "Meat", "Fish", "Sweet", "Gold"]]

#One-hot encode categorical variables
X = pd.get_dummies(X, columns=["Education", "Marital_Status"], drop_first=True)

#Initializing models
models = [
    LinearRegression(),
    Lasso(),
    Ridge(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
]

#Implementing KFold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Making predictions
for model in models:
    mse_train_list, mae_train_list, r2_train_list = [], [], []
    mse_test_list, mae_test_list, r2_test_list = [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)

        y_train_predictions = model.predict(X_train)
        y_test_predictions = model.predict(X_test)

        # Evaluating model for training set
        mse_train = mean_squared_error(y_train, y_train_predictions)
        mae_train = mean_absolute_error(y_train, y_train_predictions)
        r2_train = r2_score(y_train, y_train_predictions)

        mse_train_list.append(mse_train)
        mae_train_list.append(mae_train)
        r2_train_list.append(r2_train)

        # Evaluating model for testing set
        mse_test = mean_squared_error(y_test, y_test_predictions)
        mae_test = mean_absolute_error(y_test, y_test_predictions)
        r2_test = r2_score(y_test, y_test_predictions)

        mse_test_list.append(mse_test)
        mae_test_list.append(mae_test)
        r2_test_list.append(r2_test)

    # Printing the results for the current model
    print(f"Model: {model.__class__.__name__}")
    print(f"Average MSE for training set: {np.mean(mse_train_list)}")
    print(f"Average MAE for training set: {np.mean(mae_train_list)}")
    print(f"Average R-Squared for training set: {np.mean(r2_train_list)}\n")
    print(f"Average MSE for testing set: {np.mean(mse_test_list)}")
    print(f"Average MAE for testing set: {np.mean(mae_test_list)}")
    print(f"Average R-Squared for testing set: {np.mean(r2_test_list)}")
    print("-" * 50)
