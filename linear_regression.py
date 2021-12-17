import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import array, mod
from sklearn.linear_model import LinearRegression
import csv

time_studied = np.array([])
scores = np.array([])

with open(r'data\Student Study Hour V2.csv', mode='r') as data_file:
	line = csv.reader(data_file, delimiter="\n")
	count = 0
	for row in line:
		if not (count == 0):
			data = row[0].split(",")
			# print(float(data[0]))
			time_studied = np.append(time_studied, float(data[0]))
			scores = np.append(scores, int(data[1]))
		else:
			count += 1
#* scikit-learn requires arrays in a vertical format

model = LinearRegression()

time_studied = np.reshape(time_studied, (-1, 1))
scores = np.reshape(scores, (-1, 1))
model.fit(time_studied, scores)

plt.scatter(time_studied, scores)
plt.plot(np.linspace(0, max(time_studied), 100), model.predict(np.linspace(0, max(time_studied), 100)), 'r')
plt.ylim(0, 100)
plt.xlim(0, 10)

print(str(model.coef_) + ", " + str(model.intercept_))
plt.show()