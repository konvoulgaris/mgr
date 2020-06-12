
import os
import librosa
import numpy as np
import json

GTZAN_PATH = "data/gtzan"
TRACK_DURATION = 30
SAMPLE_DURATION = 1
EXPORT_GTZAN_PATH = "data/gtzan.npz"
EXPORT_GENRES_PATH = "data/genres.json"

# Extract MFCCs from the middle of the audio file
def process_audio(path):
	signal, sr = librosa.load(path)
	length = librosa.get_duration(signal)
	
	start = int(sr * (length / 2))
	end = int(start + (sr * SAMPLE_DURATION))

	mfcc = librosa.feature.mfcc(signal[start:end], n_mfcc=13).T
	return mfcc

# Extract MFCCs from the audio files of the GTZAN dataset
def _process_gtzan():
	features = []
	labels = []
	genres = []
	
	for i, (path, dirs, files) in enumerate(os.walk(GTZAN_PATH)):
		if path == GTZAN_PATH:
			# Directories => genres
			for genre in dirs:
				genres.append(genre)
		else:
			for file in files:
				signal, sr = librosa.load(os.path.join(path, file))

				# Extract MFCC for every second of the audio signal
				for j in range(0, TRACK_DURATION - SAMPLE_DURATION, SAMPLE_DURATION):
					start = sr * j
					end = start + (sr * SAMPLE_DURATION)
					
					mfcc = librosa.feature.mfcc(y=signal[start:end], n_mfcc=13).T
					
					features.append(mfcc)
					labels.append(i - 1)

	features = np.array(features)
	labels = np.array(labels)
	
	np.savez(EXPORT_GTZAN_PATH, features=features, labels=labels)
	
	try:
		with open(EXPORT_GENRES_PATH, "w") as file:
			json.dump(genres, file)
	except Exception:
		print(f"Failed to write {EXPORT_GENRES_PATH}")
		exit(1)

if __name__ == "__main__":
	_process_gtzan()
