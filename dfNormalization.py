import matplotlib.pyplot as plt
import seaborn as sns

from dfCreation import newDF

# print(tabulate(newDF))
nan_check_column = newDF.isna().sum()
print(nan_check_column)

# Identifying the skewness of the feature Income to handle its missing values.
sns.histplot(newDF['Income'], kde=True)
plt.show()

# Replacing the null values in the Income feature with the feature median:
newDF["Income"] = newDF["Income"].fillna(newDF["Income"].median())
print(newDF.info())  # Rechecking the DF for null values.
