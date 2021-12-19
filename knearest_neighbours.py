from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = load_breast_cancer()

#// print(data.feature_names)
#// print(data.target_names)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)
#* Take data (feature data & target data (which are the classes) and add each into numpy array), and split 4:1

clf = KNeighborsClassifier(n_neighbors=3)
#* have 2 classes, so 3 neighbours works perfectly fine

clf.fit(x_train, y_train)
# print(clf.score(x_test, y_test))
#* train, then test (i.e. evaluate) the model

#! clf.predict()
#* Can pass in np.array of all parameters and predict class based on values of parameters
#* Useful for real world application of model

#* ---------------------------------------------------------------------------------------------------------------------------- *#

data_df = pd.DataFrame(data.data, columns=data.feature_names)
# print(data.target)
# print(data_df)

mean_radius_data = np.array(data_df["mean radius"])
mean_texture_data = np.array(data_df["mean texture"])

classes = data.target

for i in range(0, len(classes)):
	plt.scatter(mean_radius_data[i], mean_texture_data[i], color=colors.get(classes[i]))

plt.show()