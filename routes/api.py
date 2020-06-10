import os
import json
import numpy as np
import io
import random
import matplotlib.pyplot as plt

from flask import Blueprint, request, g, Response
from youtube_dl import YoutubeDL
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from lib.process import process_audio

FORMAT = "mp3"
SONGS_PATH = "songs/"
GENRES_PATH = "data/genres.json"
YDL_OPT = {
	"outtmpl": SONGS_PATH + "%(id)s.%(ext)s",
	"format": "bestaudio/best",
	"postprocessors": [
		{
			"key": "FFmpegExtractAudio",
			"preferredcodec": FORMAT,
		}
	],
}
BASE_URL = "https://www.youtube.com/watch?v="
YDL = YoutubeDL(YDL_OPT)

api = Blueprint("api", __name__)


@api.route("/download", methods=["POST"])
def download():
	if not "url" in request.form:
		return { "error": "Need 'url' key in request form" }, 400

	url = request.form["url"]
	vid = url.replace(BASE_URL, "")

	YDL.download([url])
	
	return { "id": vid, "path": SONGS_PATH + vid + "." + FORMAT}, 200

@api.route("/process", methods=["POST"])
def result():
	try:
		data = json.loads(request.data)
	except Exception:
		return { "error": "Invalid request data" }, 400

	if not ("id" in data and "path" in data):
		return { "error": "Need 'id' and 'path' keys in request data" }, 400

	vid = data["id"]
	audio = process_audio(data["path"])
	
	y_nn = g.nn.predict(np.array([audio]), batch_size=32)[0]
	r_nn = []
	
	for n in y_nn:
		r_nn.append(n * 100)

	x, y = audio.shape
	audio = audio.reshape(1, x * y)
	audio = g.knn.sc.transform(audio)

	y_knn = g.knn.predict_proba(audio)[0]
	r_knn = []

	for k in y_knn:
		r_knn.append(k * 100)

	audio = g.svm.pca.transform(audio)
	
	y_svm = g.svm.predict_proba(audio)[0]
	r_svm = []
	
	for s in y_svm:
		r_svm.append(s * 100)

	prediction = { "knn": r_knn, "svm": r_svm, "nn": r_nn }

	try:
		with open(SONGS_PATH + vid + ".json", "w") as file:
			json.dump(prediction, file)
	except Exception:
		return { "error": "Failed to write metadata" }, 500
	
	return { "id": vid, "knn": int(np.argmax(y_knn)), "svm": int(np.argmax(y_svm)), "nn": int(np.argmax(y_nn)) }, 200

@api.route("/upload/<string:vid>.png", methods=["GET"])
def upload_png(vid):
	genres = None
	results = None

	try:
		with open(GENRES_PATH, "r") as file:
			genres = json.load(file)
	
		with open(SONGS_PATH + vid + ".json", "r") as file:
			results = json.load(file)
	except Exception:
		return { "error": "Failed to read metadata" }, 500
	
	fig = Figure()
	
	axis = fig.add_subplot(1, 1, 1)
	axis.plot(genres, results["knn"], label="KNN")
	axis.plot(genres, results["svm"], label="SVM")
	axis.plot(genres, results["nn"], label="NN")
	axis.set_xlabel("Genres")
	axis.set_ylabel("Confidence")
	axis.legend()
	axis.grid()

	output = io.BytesIO()
	FigureCanvasAgg(fig).print_png(output)

	return Response(output.getvalue(), mimetype="image/png")
