import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from joblib import dump

DATA_PATH = "data/gtzan.npz"
KNN_NEIGHBORS = 10
KNN_WEIGHTS = "distance"
SVM_PROBABILITY = True
KNN_PATH = "models/knn.joblib"
SVM_PATH = "models/svm.joblib"

class KNN(KNeighborsClassifier):
	def __init__(self):
		super().__init__(n_neighbors=KNN_NEIGHBORS, weights=KNN_WEIGHTS)

	def __repr__(self):
		print(f"<KNN: {self.accuracy * 100}%>")

	def _train(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		
		self.sc = StandardScaler()
		self.sc.fit(X_train)
		X_train = self.sc.transform(X_train)
		X_test = self.sc.transform(X_test)

		self.fit(X_train, y_train)
		self.accuracy = self.score(X_test, y_test)

		print(f"Trained KNN with {self.accuracy} accuracy")

class SVM(SVC):
	def __init__(self):
		super().__init__(probability=True)

	def __repr__(self):
		print(f"<SVM: {self.accuracy * 100}%>")

	def _train(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		
		self.sc = StandardScaler()
		self.sc.fit(X_train)
		X_train = self.sc.transform(X_train)
		X_test = self.sc.transform(X_test)

		self.pca = PCA(n_components=13)
		self.pca.fit(X_train)
		X_train = self.pca.transform(X_train)
		X_test = self.pca.transform(X_test)

		self.fit(X_train, y_train)
		self.accuracy = self.score(X_test, y_test)

		print(f"Trained SVM with {self.accuracy} accuracy")

if __name__ == "__main__":
	data = np.load(DATA_PATH)

	X = data["features"]
	samples, x, y = X.shape
	X = X.reshape([samples, x * y])
	
	y = data["labels"]

	knn = KNN()
	knn._train(X, y)

	svm = SVM()
	svm._train(X, y)

	dump(knn, KNN_PATH)
	dump(svm, SVM_PATH)
