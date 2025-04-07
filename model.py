import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Function to extract features from audio file
def extract_features(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Load dataset
def load_dataset(folder_pos, folder_neg):
    X = [] # Positive class
    y = [] # Negative class
    for file in os.listdir(folder_pos):
        if file.endswith('.wav'):
            features = extract_features(os.path.join(folder_pos, file))
            X.append(features)
            y.append(1) 
    for file in os.listdir(folder_neg):
        if file.endswith('.wav'):
            features = extract_features(os.path.join(folder_neg, file))
            X.append(features)
            y.append(0)
    return np.array(X), np.array(y)

# Predicts audio classification
def predict_audio_class(file_path):
    clf = joblib.load('audio_classifier.pkl')

    features = extract_features(file_path)
    features = features.reshape(1, -1)
    prediction = clf.predict(features)
    
    if prediction[0] == 1:
        return True
    else:
        return False

if __name__ == "__main__":
    pos_folder = "data/fart_dataset"
    neg_folder = "data/music_dataset/Trumpet"

    X, y = load_dataset(pos_folder, neg_folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(clf, 'audio_classifier.pkl')