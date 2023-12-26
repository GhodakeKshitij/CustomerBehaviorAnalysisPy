from datetime import datetime
from tabulate import tabulate
from dfCreation import newDF
import pandas as pd

print(newDF.columns)
# Finding out the age of each customer:
Cust_Age = []
Birth_Year = newDF["Birth_Year"]
for i in Birth_Year:
    age = 2023 - i
    Cust_Age.append(age)
# print(Cust_Age)

# Extracting the customer joining year from the customer joining date:
Cust_Joining_Year = []
No_Of_Years = []

# Convert "Cust_Joining_Date" to datetime
Cust_Joining_Date = pd.to_datetime(newDF["Cust_Joining_Date"], format="%d%m%y")

# Extracting the year from the joining date and calculating the tenure
for date in Cust_Joining_Date:
    joining_year = date.year
    current_year = datetime.now().year
    years_with_store = current_year - joining_year
    No_Of_Years.append(years_with_store)

# Printing the result
# print(tabulate(No_Of_Years))
print(len(No_Of_Years))

newDF.insert(2, "Cust_Age", Cust_Age)
newDF.insert(9, "Cust_Tenure", No_Of_Years)
# print(tabulate(newDF))
print(newDF.columns)

# Updating the values of col "Campaign5" according to col "Response"
for index, row in newDF.iterrows():
    if row["Response"] >= 1:
        if newDF.at[index, "Campaign_5"] == 1:
            continue
        else:
            newDF.at[index, "Campaign_5"] = row["Campaign_5"] + 1

for index, row in newDF.iterrows():
    if row["Campaign_5"] >= 1:
        if newDF.at[index, "Response"] == 1:
            continue
        else:
            newDF.at[index, "Response"] == row["Response"] + 1
print(tabulate(newDF))

# Removing the columns which are not useful:
finalDF = pd.DataFrame(newDF.drop(["Z_CostContact", "Z_Revenue"], axis=1))
finalDF = finalDF.rename(
    columns={
        "Meat_Products": "Meat",
        "Fish_Products": "Fish",
        "Gold_Products": "Gold",
        "Sweet_Products": "Sweet",
    }
)
finalDF.to_csv(
    "E:\\MCS\\SYMCS\\Semester 3\\ML\\Project\\CustomerBehaviorAnalysisPy\\finalDataframe.csv",
    index=False,
)
