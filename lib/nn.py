import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow import keras

DATA_PATH = "data/gtzan.npz"
NN_PATH = "models/nn.h5"

if __name__ == "__main__":
	data = np.load(DATA_PATH)

	X = data["features"]
	y = data["labels"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

		keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
		keras.layers.Dropout(0.3),

		keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
		keras.layers.Dropout(0.3),

		keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
		keras.layers.Dropout(0.3),

		keras.layers.Dense(10, activation='softmax')
	])

	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
	model.summary()
	model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), epochs=75)

	model.save(NN_PATH)
