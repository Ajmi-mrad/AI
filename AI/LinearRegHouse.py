import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Download and load the dataset
path = kagglehub.dataset_download("justinas/housing-in-london")
file_path = path + '/housing_in_london_monthly_variables.csv'

data = pd.read_csv(file_path)

data.info()
print(data.head())

data = data[['average_price', 'houses_sold']]

# nfaskh null lkol 
data = data.dropna()

# Feature Selection
X = data[['houses_sold']]  
y = data['average_price']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['houses_sold'], y=y_test, color='blue', label='Actual')
sns.scatterplot(x=X_test['houses_sold'], y=y_pred, color='red', label='Predicted')
plt.title('Linear Reg: Predicted vs Actual Housing Prices')
plt.xlabel('Houses Sold')
plt.ylabel('Average Price')
plt.legend()
plt.show()