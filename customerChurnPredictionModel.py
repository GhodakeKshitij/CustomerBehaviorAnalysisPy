import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, \
    mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import featureEngineering

#Accessing relevant features
X = featureEngineering.finalDF[
    ["Birth_Year", "Cust_Age", "Education", "Marital_Status", "Income", "No_Of_Kids", "No_Of_Teens",
     "Cust_Tenure", "Days_Since_Last_Purchase", "Wines", "Fruits", "Meat", "Fish",
     "Sweet", "Gold", "Deal_Purchases", "Web_Purchases", "Catalog_Purchases", "Store_Purchases",
     "Web_Visits_Per_Month", "Campaign_1", "Campaign_2", "Campaign_3", "Campaign_4", "Campaign_5"]]

# One-hot encoding
X = pd.get_dummies(X, columns=["Education", "Marital_Status"], drop_first=True)
X["Income"] = X["Income"].fillna(X["Income"].median())
X["Income"] = X["Income"].astype(np.int64)

#Scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y_churn = featureEngineering.finalDF[["Response"]].values

models = [
    RandomForestClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
    SVC(random_state=42),
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=42)
]

#Implementing KFold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Making Predictions
for model in models:
    mse_train_list, mae_train_list, r2_train_list = [], [], []
    mse_test_list, mae_test_list, r2_test_list = [], [], []
    accuracy_train_list, accuracy_test_list = [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_churn[train_index], y_churn[test_index]

        model.fit(X_train, y_train.ravel())
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        #Evaluating results for Training set
        mse_train = mean_squared_error(y_train.ravel(), y_train_pred)
        mae_train = mean_absolute_error(y_train.ravel(), y_train_pred)
        r2_train = r2_score(y_train.ravel(), y_train_pred)
        accuracy_train = accuracy_score(y_train.ravel(), y_train_pred)

        mse_train_list.append(mse_train)
        mae_train_list.append(mae_train)
        r2_train_list.append(r2_train)
        accuracy_train_list.append(accuracy_train)

        #Evaluating results for Testing set
        mse_test = mean_squared_error(y_test.ravel(), y_test_pred)
        mae_test = mean_absolute_error(y_test.ravel(), y_test_pred)
        r2_test = r2_score(y_test.ravel(), y_test_pred)
        accuracy_test = accuracy_score(y_test.ravel(), y_test_pred)

        mse_test_list.append(mse_test)
        mae_test_list.append(mae_test)
        r2_test_list.append(r2_test)

    #Printing the results
    print(f"Model: {model.__class__.__name__}")
    print(f"Average MSE for training set: {np.mean(mse_train_list)}")
    print(f"Average MAE for training set: {np.mean(mae_train_list)}")
    print(f"Average R-Squared for training set: {np.mean(r2_train_list)}\n")
    print(f"Average MSE for test set: {np.mean(mse_test_list)}")
    print(f"Average MAE for test set: {np.mean(mae_test_list)}")
    print(f"Average R-Squared for test set: {np.mean(r2_test_list)}\n")
    print("Accuracy:")
    print(f"{accuracy_score(y_test.ravel(), y_test_pred)}")
    print(f"Classification report:")
    print(f"{classification_report(y_test.ravel(), y_test_pred, zero_division=1)}")
    print(f"Confusion matrix:")
    print(f"{confusion_matrix(y_test.ravel(), y_test_pred)}")
    print("-" * 70)
