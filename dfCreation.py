import pandas as pd
from tabulate import tabulate

# Load the DataFrame
data = pd.read_excel(
    "E:\MCS\SYMCS\Semester 3\ML\CustomerBehaviorAnalysisPy\marketing_campaign.xlsx"
)

feature_names = [
    "ID\tYear_Birth\tEducation\tMarital_Status\tIncome\tKidhome\tTeenhome\tDt_Customer\tRecency\tMntWines\tMntFruits\tMntMeatProducts\tMntFishProducts\tMntSweetProducts\tMntGoldProds\tNumDealsPurchases\tNumWebPurchases\tNumCatalogPurchases\tNumStorePurchases\tNumWebVisitsMonth\tAcceptedCmp3\tAcceptedCmp4\tAcceptedCmp5\tAcceptedCmp1\tAcceptedCmp2\tComplain\tZ_CostContact\tZ_Revenue\tResponse"
]

# Create empty lists to store values
(
    ID,
    Birth_Year,
    Education,
    Marital_Status,
    Income,
    No_Of_Kids,
    No_Of_Teens,
    Customer_Joining_Date,
    Recency,
    Wines,
    Fruits,
    Meat,
    Fish,
    Sweet,
    Gold,
    NumDealsPurchases,
    NumWebPurchases,
    NumCatalogPurchases,
    NumStorePurchases,
    NumWebVisitsMonth,
    AcceptedCamp1,
    AcceptedCamp2,
    AcceptedCamp3,
    AcceptedCamp4,
    AcceptedCamp5,
    Complain,
    Z_CostContact,
    Z_Revenue,
    Response,
) = ([] for _ in range(29))

# Iterate over the rows of the DataFrame
for index, row in data.iterrows():
    ID.append(row["ID"])
    Birth_Year.append(row["Year_Birth"])
    Education.append(row["Education"])
    Marital_Status.append(row["Marital_Status"])
    Income.append(row["Income"])
    No_Of_Kids.append(row["Kidhome"])
    No_Of_Teens.append(row["Teenhome"])
    Customer_Joining_Date.append(row["Dt_Customer"])
    Recency.append(row["Recency"])
    Wines.append(row["MntWines"])
    Fruits.append(row["MntFruits"])
    Meat.append(row["MntMeatProducts"])
    Fish.append(row["MntFishProducts"])
    Sweet.append(row["MntSweetProducts"])
    Gold.append(row["MntGoldProds"])
    NumDealsPurchases.append(row["NumDealsPurchases"])
    NumWebPurchases.append(row["NumWebPurchases"])
    NumCatalogPurchases.append(row["NumCatalogPurchases"])
    NumStorePurchases.append(row["NumStorePurchases"])
    NumWebVisitsMonth.append(row["NumWebVisitsMonth"])
    AcceptedCamp1.append(row["AcceptedCmp1"])
    AcceptedCamp2.append(row["AcceptedCmp2"])
    AcceptedCamp3.append(row["AcceptedCmp3"])
    AcceptedCamp4.append(row["AcceptedCmp4"])
    AcceptedCamp5.append(row["AcceptedCmp5"])
    Complain.append(row["Complain"])
    Z_CostContact.append(row["Z_CostContact"])
    Z_Revenue.append(row["Z_Revenue"])
    Response.append(row["Response"])

# Create a new DataFrame using the lists
newDF = pd.DataFrame(
    {
        "ID": ID,
        "Birth_Year": Birth_Year,
        "Education": Education,
        "Marital_Status": Marital_Status,
        "Income": Income,
        "No_Of_Kids": No_Of_Kids,
        "No_Of_Teens": No_Of_Teens,
        "Cust_Joining_Date": Customer_Joining_Date,
        "Days_Since_Last_Purchase": Recency,
        "Wines": Wines,
        "Fruits": Fruits,
        "Meat_Products": Meat,
        "Fish_Products": Fish,
        "Sweet_Products": Sweet,
        "Gold_Products": Gold,
        "Deal_Purchases": NumDealsPurchases,
        "Web_Purchases": NumWebPurchases,
        "Catalog_Purchases": NumCatalogPurchases,
        "Store_Purchases": NumStorePurchases,
        "Web_Visits_Per_Month": NumWebVisitsMonth,
        "Campaign_1": AcceptedCamp1,
        "Campaign_2": AcceptedCamp2,
        "Campaign_3": AcceptedCamp3,
        "Campaign_4": AcceptedCamp4,
        "Campaign_5": AcceptedCamp5,
        "Complain": Complain,
        "Z_CostContact": Z_CostContact,
        "Z_Revenue": Z_Revenue,
        "Response": Response,
    }
)

print(tabulate(newDF))
print(newDF.columns)
