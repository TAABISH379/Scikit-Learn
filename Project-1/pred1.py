import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv("Student_Performance.csv")

x= data[['Hours Studied']]
y=data['Performance Index']

model= LinearRegression()
model.fit(x,y)

predicted_score = model.predict(x)

mae= mean_absolute_error(y, predicted_score)

mse= mean_squared_error(y, predicted_score)

rmse = np.sqrt(mse)

print("Mean absolute error: ", mae)

print("Mean squared error: ", mse)

print("Root MSE: ", rmse)

plt.figure(figsize=(20,15))
plt.hist(data["Performance Index"], bins=30, color='blue', edgecolor='black')
plt.title("Distribution of Final Exam Scores")
plt.xlabel("Scores ")
plt.ylabel("Time studied in hrs")
plt.grid(True)
plt.show()


# Take input from user
hours = float(input("Enter hours studied: "))

# Convert to 2D
hours_array = [[hours]]

# Predict
predicted_new_score = model.predict(hours_array)

# Output
print(f"Predicted Performance Index: {predicted_new_score[0]:.2f}")