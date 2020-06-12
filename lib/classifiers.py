import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from joblib import dump

DATA_PATH = "data/gtzan.npz"
EXPORT_KNN_PATH = "models/knn.joblib"
EXPORT_SVM_PATH = "models/svm.joblib"

class KNN(KNeighborsClassifier):
	def __init__(self):
		super().__init__(n_neighbors=10, weights="distance")

	def __repr__(self):
		print(f"<KNN: {self.accuracy * 100}%>")

	def _train(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		
		# Scale data
		self.sc = StandardScaler()
		self.sc.fit(X_train)
		X_train = self.sc.transform(X_train)
		X_test = self.sc.transform(X_test)

		# Generate model
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
		
		# Scale data
		self.sc = StandardScaler()
		self.sc.fit(X_train)
		X_train = self.sc.transform(X_train)
		X_test = self.sc.transform(X_test)

		# Apply PCA
		self.pca = PCA(n_components=13)
		self.pca.fit(X_train)
		X_train = self.pca.transform(X_train)
		X_test = self.pca.transform(X_test)

		# Generate model
		self.fit(X_train, y_train)
		self.accuracy = self.score(X_test, y_test)

		print(f"Trained SVM with {self.accuracy} accuracy")

if __name__ == "__main__":
	# Load data
	data = np.load(DATA_PATH)

	X = data["features"]
	samples, x, y = X.shape
	X = X.reshape([samples, x * y])
	
	y = data["labels"]

	# Generate KNN
	knn = KNN()
	knn._train(X, y)

	# Export KNN
	dump(knn, EXPORT_KNN_PATH)

	# Generate SVM
	svm = SVM()
	svm._train(X, y)

	# Export SVM
	dump(svm, EXPORT_SVM_PATH)
