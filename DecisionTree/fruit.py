from sklearn.tree import DecisionTreeClassifier

x=[
    [7,9],
    [8,3],
    [5,10],
    [2,2]
]

y=[0, 0, 1, 1]

model=DecisionTreeClassifier()

model.fit(x,y)

size=float(input("Enter the size of the fruit in cms: "))
shade=float(input("Enter the clour shade of the fruit on scale(0-10): "))

result=model.predict([[size,shade]])[0]

if result ==0:
    print("This is likely an apple")

else:
    print("This is likely an orange")