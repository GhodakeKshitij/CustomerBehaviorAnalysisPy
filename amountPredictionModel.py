from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from featureEngineering import finalDF

# Selecting features and target variable
X = finalDF[["Cust_Age", "Education", "Marital_Status", "Income"]]
y = finalDF[["Wines", "Fruits", "Meat", "Fish", "Sweet", "Gold"]]

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=["Marital_Status", "Education"], drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handling missing values in target variable
y_train = y_train.dropna()

# Drop rows with NaN in X_train and X_test
X_train = pd.DataFrame(X_train).dropna()
X_test = pd.DataFrame(X_test).dropna()

# Align y_train with the corresponding rows in X_train
y_train = y_train.loc[X_train.index]

# Standardizing and clipping values in features for X_train
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_train = np.clip(X_train, -np.finfo(np.float64).max, np.finfo(np.float64).max)

# Align y_test with the corresponding rows in X_test
y_test = y_test.loc[X_test.index]

# Standardizing and clipping values in features for X_test
X_test = scalar.transform(X_test)
X_test = np.clip(X_test, -np.finfo(np.float64).max, np.finfo(np.float64).max)

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
