import os
import json

from flask import Flask, render_template, request, g, Response
from joblib import load
from tensorflow import keras

from lib.classifiers import KNN, SVM

MGR_HOST = os.getenv("MGR_HOST", "0.0.0.0")
MGR_PORT = os.getenv("PORT", "5000")

from api import api

app = Flask(__name__)
app.register_blueprint(api, url_prefix="/api")

app.config['UPLOAD_FOLDER'] = "uploads"

@app.route("/", methods=["GET"])
def index():
	return render_template("index.html")

@app.route("/upload", methods=["GET"])
def upload():
	if not ("id" in request.args and "knn" in request.args and "svm" in
			request.args and "nn" in request.args):
		return { "error": "Need 'id', 'knn', 'svm' and 'nn' keys in request arguments" }, 400

	vid = request.args["id"]
	knn = request.args["knn"]
	svm = request.args["svm"]
	nn = request.args["nn"]
	genres = None
	results = None

	try:
		with open("data/genres.json", "r") as file:
			genres = json.load(file)
	
		with open("songs/" + vid + ".json", "r") as file:
			results = json.load(file)
	except Exception:
		return { "error": "Failed to read metadata" }, 500

	return render_template("upload.html", vid=vid, knn=knn, svm=svm, nn=nn, genres=genres, results=results)

@app.before_request
def before_request():
	if not hasattr(g, "knn"):
		g.knn = load("models/knn.joblib")
		g.svm = load("models/svm.joblib")
		g.nn = keras.models.load_model("models/nn.h5")
		g.genres = json.load(open("data/genres.json", "r"))

if __name__ == "__main__":
	app.run(MGR_HOST, MGR_PORT, debug=True)
