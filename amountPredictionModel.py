import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
from featureEngineering import finalDF

data = finalDF.head(1000)
print(tabulate(data))
print(data.info())

# Handling NaN values.
data['Income'] = data['Income'].fillna(data['Income'].median())
data['Income'] = data['Income'].astype(np.int64)
print(data.info())

# Selecting features and target variable
X = data[["Cust_Age", "Education", "Marital_Status", "Income"]]
y = data[["Wines", "Fruits", "Meat", "Fish", "Sweet", "Gold"]]

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=["Education", "Marital_Status"], drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initializing models
models = [
    LinearRegression(),
    Lasso(),
    Ridge(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
]

# Training and evaluating models
for model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions, multioutput="raw_values")

    # Printing the results
    model_name = model.__class__.__name__
    print(f"Model: {model_name}")
    for i, product in enumerate(y.columns):
        # To print the individual prediction values for each product
        # print(f"Predictions for {product}")
        # print(predictions[:, i])
        # print()
        # print(f"Actual values for {product}")
        # print(y_test[product].values)
        # print()
        print(f"Mean Squared Error for {product}: {mse[i]}")
    print("\n")
