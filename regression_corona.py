import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("covid19data_cali.csv", sep=",")

data = data[["Total Count Confirmed", "Total Count Deaths", "COVID-19 Positive Patients", "Suspected COVID-19 Positive Patients", "ICU COVID-19 Positive Patients", "ICU COVID-19 Suspected Patients"]]
data = data.dropna()
print(data.head)
# Label (what we want to get from the attributes)
predict = "Total Count Confirmed"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.01)

linear = linear_model.LinearRegression()
# Create best fit line
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print("Accuracy ", accuracy)

print('Coefficient: \n', linear.coef_) # These are each slope value
print('Intercept: \n', linear.intercept_) # This is the intercept

predictions = linear.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "Total Count Deaths"  # Attribute

style.use("seaborn")
pyplot.scatter(data[p], data["Total Count Confirmed"])
pyplot.ylabel("Total Count Confirmed")
pyplot.xlabel(p)
pyplot.show()