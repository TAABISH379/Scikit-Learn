from sklearn.neighbors import KNeighborsClassifier

x=[
    [100, 7],
    [200, 7.5],
    [389, 8.1],
    [405, 8.0],
    [500, 9.5]
]

# 0-> Apple, 1-> Banana

y= [0, 0, 0, 1, 1]

model= KNeighborsClassifier(n_neighbors=3)

model.fit(x,y)

weight = float(input("Enter the weight in gms: "))

size = float(input("Enter the size in cms: "))

prediction= model.predict([[weight, size]])[0]

if prediction == 0:
    print("This is likely an Apple")

else:
    print("This is likely an Orange")