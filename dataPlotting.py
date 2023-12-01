from featureEngineering import finalDF
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

unique_values = finalDF["Education"].unique()
# print(unique_values)

# Plotting bar graph for the count of education type:
for column in finalDF.columns:
    if finalDF[column].dtype == "object":
        # Count the unique values in the column
        value_counts = finalDF[column].value_counts()
        # Plot the unique values as a bar plot
        plot = value_counts.plot(kind="bar", figsize=(10, 5))
        # Set labels and title
        plt.xlabel(column)
        plot.set_xticklabels(
            plot.get_xticklabels(), rotation=0
        )  # Adjust the rotation angle if needed

        plt.ylabel("Count")
        plt.title(f"Unique Values in {column}")
        plt.show()

# Correlation Matrix:
data = {
    "Age": finalDF["Cust_Age"],
    "Education": finalDF["Education"],
    "Marital_Status": finalDF["Marital_Status"],
    "Income": finalDF["Income"],
    "No_Of_Kids": finalDF["No_Of_Kids"],
    "No_Of_Teens": finalDF["No_Of_Teens"],
    "Cust_Tenure": finalDF["Cust_Tenure"],
    "Days_Since_Last_Purchase": finalDF["Days_Since_Last_Purchase"],
    "Wines": finalDF["Wines"],
    "Fruits": finalDF["Fruits"],
    "Meat": finalDF["Meat"],
    "Fish": finalDF["Fish"],
    "Sweet": finalDF["Sweet"],
    "Gold": finalDF["Gold"],
    "Deal_Purchases": finalDF["Deal_Purchases"],
    "Web_Purchases": finalDF["Web_Purchases"],
    "Catalog_Purchases": finalDF["Catalog_Purchases"],
    "Store_Purchases": finalDF["Store_Purchases"],
    "Web_Visits_Per_Month": finalDF["Web_Visits_Per_Month"],
    "Camp_1": finalDF["Campaign_1"],
    "Camp_2": finalDF["Campaign_2"],
    "Camp_3": finalDF["Campaign_3"],
    "Camp_4": finalDF["Campaign_4"],
    "Camp_5": finalDF["Campaign_5"],
}
df = pd.DataFrame(data)
corr_matrix = df.corr()
# print(tabulate(corr_matrix))
corr_matrix.to_csv(
    "E:\MCS\SYMCS\Semester 3\ML\CustomerBehaviorAnalysisPy\corrMatrix.csv",
    index=False,
)
corr_HM = sns.heatmap(
    corr_matrix,
    vmin=-1,
    vmax=1,
    center=0,
    cmap=sns.color_palette("coolwarm"),
    square=True
)
plt.title("Correlation between all the features.")
plt.show()

# Checking the highly correlated variables from the heatmap:
threshold = 0.5
# high_corr = corr_matrix.abs() > threshold
correlated_pairs = [
    (col1, col2)
    for col1 in corr_matrix.columns
    for col2 in corr_matrix.columns
    if corr_matrix.loc[col1, col2] > threshold and col1 != col2
]
print(correlated_pairs)

# Plotting correlation between multiple variables like Income, Wines, Meat, Fish, Fruits, Catalog_Purchases, Store_Purchases:
data1 = {
    "Income": finalDF["Income"],
    "Wines": finalDF["Wines"],
    "Meat": finalDF["Meat"],
    "Fish": finalDF["Fish"],
    "Fruits": finalDF["Fruits"],
    "Sweet": finalDF["Sweet"],
    "Catalog_Purchases": finalDF["Catalog_Purchases"],
    "Store_Purchases": finalDF["Store_Purchases"],
}
data1 = pd.DataFrame(data1)
data1_corr_matrix = data1.corr().round(2)
dat1HM = sns.heatmap(
    data1_corr_matrix, vmin=0, vmax=1, cmap="coolwarm", square=True, annot=True
)
plt.title("Correlation between multiple variables.")
plt.show()
