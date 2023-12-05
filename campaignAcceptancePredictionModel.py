import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from featureEngineering import finalDF

# Selecting only the campaign-related features
X = finalDF[["Campaign_1", "Campaign_2", "Campaign_3", "Campaign_4", "Campaign_5"]]

# Creating binary target variables for each campaign
y_campaign_1 = finalDF["Campaign_1"]
y_campaign_2 = finalDF["Campaign_2"]
y_campaign_3 = finalDF["Campaign_3"]
y_campaign_4 = finalDF["Campaign_4"]
y_campaign_5 = finalDF["Campaign_5"]

# Split the data for each campaign
(
    X_train_campaign_1,
    X_test_campaign_1,
    y_train_campaign_1,
    y_test_campaign_1,
) = train_test_split(X, y_campaign_1, test_size=0.2, random_state=42)
(
    X_train_campaign_2,
    X_test_campaign_2,
    y_train_campaign_2,
    y_test_campaign_2,
) = train_test_split(X, y_campaign_2, test_size=0.2, random_state=42)
(
    X_train_campaign_3,
    X_test_campaign_3,
    y_train_campaign_3,
    y_test_campaign_3,
) = train_test_split(X, y_campaign_3, test_size=0.2, random_state=42)
(
    X_train_campaign_4,
    X_test_campaign_4,
    y_train_campaign_4,
    y_test_campaign_4,
) = train_test_split(X, y_campaign_4, test_size=0.2, random_state=42)
(
    X_train_campaign_5,
    X_test_campaign_5,
    y_train_campaign_5,
    y_test_campaign_5,
) = train_test_split(X, y_campaign_5, test_size=0.2, random_state=42)

# Choose a classification algorithm (Random Forest in this example)
classifier_campaign_1 = RandomForestClassifier()
classifier_campaign_2 = RandomForestClassifier()
classifier_campaign_3 = RandomForestClassifier()
classifier_campaign_4 = RandomForestClassifier()
classifier_campaign_5 = RandomForestClassifier()

# Train the models for each campaign
classifier_campaign_1.fit(X_train_campaign_1, y_train_campaign_1)
classifier_campaign_2.fit(X_train_campaign_2, y_train_campaign_2)
classifier_campaign_3.fit(X_train_campaign_3, y_train_campaign_3)
classifier_campaign_4.fit(X_train_campaign_4, y_train_campaign_4)
classifier_campaign_5.fit(X_train_campaign_5, y_train_campaign_5)

# Make predictions on the test sets for each campaign
predictions_campaign_1 = classifier_campaign_1.predict(X_test_campaign_1)
predictions_campaign_2 = classifier_campaign_2.predict(X_test_campaign_2)
predictions_campaign_3 = classifier_campaign_3.predict(X_test_campaign_3)
predictions_campaign_4 = classifier_campaign_4.predict(X_test_campaign_4)
predictions_campaign_5 = classifier_campaign_5.predict(X_test_campaign_5)

# Evaluate the models for each campaign (optional)
accuracy_campaign_1 = accuracy_score(y_test_campaign_1, predictions_campaign_1)
print(f"Accuracy Campaign 1: {accuracy_campaign_1}")
accuracy_campaign_2 = accuracy_score(y_test_campaign_2, predictions_campaign_2)
print(f"Accuracy Campaign 2: {accuracy_campaign_2}")
accuracy_campaign_3 = accuracy_score(y_test_campaign_3, predictions_campaign_3)
print(f"Accuracy Campaign 3: {accuracy_campaign_3}")
accuracy_campaign_4 = accuracy_score(y_test_campaign_4, predictions_campaign_4)
print(f"Accuracy Campaign 4: {accuracy_campaign_4}")
accuracy_campaign_5 = accuracy_score(y_test_campaign_5, predictions_campaign_5)
print(f"Accuracy Campaign 5: {accuracy_campaign_5}")

print("Campaign 1 Metrics:")
print(classification_report(y_test_campaign_1, predictions_campaign_1))
print(confusion_matrix(y_test_campaign_1, predictions_campaign_1))

print("Campaign 2 Metrics:")
print(classification_report(y_test_campaign_2, predictions_campaign_2))
print(confusion_matrix(y_test_campaign_2, predictions_campaign_2))

print("Campaign 3 Metrics:")
print(classification_report(y_test_campaign_3, predictions_campaign_3))
print(confusion_matrix(y_test_campaign_3, predictions_campaign_3))

print("Campaign 4 Metrics:")
print(classification_report(y_test_campaign_4, predictions_campaign_4))
print(confusion_matrix(y_test_campaign_4, predictions_campaign_4))

print("Campaign 5 Metrics:")
print(classification_report(y_test_campaign_5, predictions_campaign_5))
print(confusion_matrix(y_test_campaign_5, predictions_campaign_5))

# Make predictions for existing data
existing_predictions_campaign_1 = classifier_campaign_1.predict(X)
existing_predictions_campaign_2 = classifier_campaign_2.predict(X)
existing_predictions_campaign_3 = classifier_campaign_3.predict(X)
existing_predictions_campaign_4 = classifier_campaign_4.predict(X)
existing_predictions_campaign_5 = classifier_campaign_5.predict(X)
