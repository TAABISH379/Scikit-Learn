import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Salary_dataset.csv")

x= data[['YearsExperience']]
y= data['Salary']

model = LinearRegression()

model.fit(x,y)

exp=float(input("Enter your experience in years: "))

pred_salary= model.predict([[exp]])

print(f"Based on your experience {exp} years you can earn upto rupees {pred_salary[0]} per month")