
import os
import librosa
import numpy as np
import json

GTZAN_PATH = "data/gtzan"
TRACK_DURATION = 30 
SAMPLE_DURATION = 1
TRACK_SAMPLING_RATE = 22050
EXPORT_PATH = "data/gtzan.npz"
GENRES_PATH = "data/genres.json"

def process_audio(path):
	signal = librosa.load(path)[0]
	length = librosa.get_duration(signal)
	
	start = int(TRACK_SAMPLING_RATE * (length / 2))
	end = int(start + (TRACK_SAMPLING_RATE * SAMPLE_DURATION))

	mfcc = librosa.feature.mfcc(signal[start:end], n_mfcc=13)
	mfcc = mfcc.T
	return mfcc

def _process_gtzan():
	features = []
	labels = []
	genres = []
	
	for i, (path, dirs, files) in enumerate(os.walk(GTZAN_PATH)):
		if path == GTZAN_PATH:
			for genre in dirs:
				genres.append(genre)
		else:
			for file in files:
				signal = librosa.load(os.path.join(path, file))[0]
				
				for j in range(0, TRACK_DURATION - SAMPLE_DURATION, SAMPLE_DURATION):
					start = TRACK_SAMPLING_RATE * j
					end = start + (TRACK_SAMPLING_RATE * SAMPLE_DURATION)
					
					mfcc = librosa.feature.mfcc(y=signal[start:end], n_mfcc=13)
					mfcc = mfcc.T
					
					features.append(mfcc)
					labels.append(i - 1)

	features = np.array(features)
	labels = np.array(labels)
	
	np.savez(EXPORT_PATH, features=features, labels=labels)
	
	try:
		with open(GENRES_PATH, "w") as file:
			json.dump(genres, file)
	except Exception:
		print(f"Failed to write {GENRES_PATH}")
		exit(1)

if __name__ == "__main__":
	_process_gtzan()
