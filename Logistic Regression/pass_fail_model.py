from sklearn.linear_model import LogisticRegression

x= [[1], [2], [3], [7], [10]] #hours studies
y= [0, 0, 1, 1, 1] #p-1 f-0

model = LogisticRegression()

model.fit(x, y)

hours = float(input("Enter how many hours you studied: "))

result = model.predict([[hours]])

if result == 1:
    print(f"Based on the hours {hours} you studied, you are likely to PASS")

else:
    print(f"Based on the hours {hours} you studied, you are likely to FAIL")