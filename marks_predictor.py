from sklearn.linear_model import LinearRegression

x= [[1], [2], [3], [4], [5]]
y=[40, 50, 66, 74, 83]

model= LinearRegression()

model.fit(x,y)

hours=float(input("Enter how many hours you studies: "))

predicted_marks = model.predict([[hours]])

print(f"Based on you hours {hours} you studied you may score around {predicted_marks}")