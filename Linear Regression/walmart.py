import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv("Walmart_Sales.csv")

data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week
data = data.drop('Date', axis=1)

data = pd.get_dummies(data, columns=['Store'], drop_first=True)

X = data.drop('Weekly_Sales', axis=1)
y = data['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

store = int(input("Enter store ID: "))
holiday = int(input("Enter number of holiday: "))
temp = float(input("Enter store temperature: "))
fuel = float(input("Enter fuel cost: "))
cpi = float(input("Enter cpi: "))
unemp = float(input("Enter unep: "))
year = int(input("Enter year: "))
month = int(input("Enter month: "))
week = int(input("Enter week: "))

input_data = pd.DataFrame([{
    'Holiday_Flag': holiday,
    'Temperature': temp,
    'Fuel_Price': fuel,
    'CPI': cpi,
    'Unemployment': unemp,
    'Year': year,
    'Month': month,
    'Week': week
}])

for col in X.columns:
    if col.startswith("Store_"):
        input_data[col] = 0

col_name = f"Store_{store}"
if col_name in input_data.columns:
    input_data[col_name] = 1

input_data = input_data.reindex(columns=X.columns, fill_value=0)

print(model.predict(input_data)[0])