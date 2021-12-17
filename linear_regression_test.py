import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import csv

from utils import file

read_grades, write_grades = file.openCSV("StudentsPerformance")

#? model = LinearRegression()
read_grades = np.reshape(read_grades, (-1, 1))
write_grades = np.reshape(write_grades, (-1, 1))
#? model.fit(read_grades, write_grades)

read_train, read_test, write_train, write_test = train_test_split(read_grades, write_grades, test_size=0.3)
model = LinearRegression()
model.fit(read_train, write_train)

plt.scatter(read_train, write_train)
plt.plot(np.linspace(0, max(read_grades), 100), model.predict(np.linspace(0, max(read_grades), 100)), 'r')
plt.xlim(0, 110)
plt.ylim(0, 110)
plt.show()


print(model.score(read_test, write_test))
