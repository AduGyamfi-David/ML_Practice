from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #* Support Vector Classifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = load_breast_cancer()

X = data.data
Y = data.target

SVC_SCORES = []
KNN_SCORES = []

for i in range(20, 70):
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
	#* random_state parameter allows for same split of data, hence allows for reproducability of results.

	svc_clf = SVC(kernel="linear", C=3)
	#* can also use kernel="rbf" (radial basis function, is the default) and kernel="polynomial", which are more resource-heavy and time costly.
	#* C ≙ number of points to misclassify in soft margins 

	svc_clf.fit(x_train, y_train)

	#* ----------------------------------------------------------------------------------- *#
	#% COMPARE WITH KNEIGHBOURS CLASSIFIER

	knn_clf = KNeighborsClassifier()
	knn_clf.fit(x_train, y_train)


	SVC_SCORES.append(svc_clf.score(x_test, y_test))
	plt.scatter(i, SVC_SCORES[i - 20], color="red")

	KNN_SCORES.append(knn_clf.score(x_test, y_test))
	plt.scatter(i, KNN_SCORES[i - 20], color="green")
	print(i)

plt.show()



# print(str(svc_clf.score(x_test, y_test)) + ", " + str(knn_clf.score(x_test, y_test)))