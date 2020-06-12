import json

from flask import Flask, render_template, request, g, Response
from joblib import load
from tensorflow import keras

from lib.classifiers import KNN, SVM
from routes.api import api

HOST = "0.0.0.0"
PORT = 5000
KNN_PATH = "models/knn.joblib"
SVM_PATH = "models/svm.joblib"
NN_PATH = "models/nn.h5"
GENRES_PATH = "data/genres.json"
SONGS_PATH = "songs/"

# Create flask app
app = Flask(__name__)

# Connect flask blueprints
app.register_blueprint(api, url_prefix="/api")

@app.route("/", methods=["GET"])
def index():
	return render_template("index.html")

@app.route("/upload", methods=["GET"])
def upload():
	if not ("id" in request.args and "knn" in request.args and "svm" in request.args and "nn" in request.args):
		return { "error": "Need 'id', 'knn', 'svm' and 'nn' keys in request arguments" }, 400

	vid = request.args["id"]
	knn = request.args["knn"]
	svm = request.args["svm"]
	nn = request.args["nn"]

	results = None

	try:
		with open(SONGS_PATH + vid + ".json", "r") as file:
			results = json.load(file)
	except Exception:
		return { "error": "Failed to read metadata" }, 500

	return render_template("upload.html", vid=vid, knn=knn, svm=svm, nn=nn, genres=g.genres, results=results)

@app.before_request
def before_request():
	if not hasattr(g, "knn"):
		g.knn = load(KNN_PATH)
		g.svm = load(SVM_PATH)
		g.nn = keras.models.load_model(NN_PATH)
		
		try:
			with open(GENRES_PATH, "r") as file:
				g.genres = json.load(file)
		except Exception:
			return { "error": "Failed to read metadata" }, 500

if __name__ == "__main__":
	app.run(HOST, PORT, debug=True)
