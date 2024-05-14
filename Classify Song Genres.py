# train classifier to distinguish between Hip-Hop and Rock  
# predict whether a song's genre can be correctly classified based on features such as danceability, energy, acousticness, and tempo, etc. 

import pandas as pd
from sklearn.model_selection import train_test_split

tracks = pd.read_csv("datasets/fma-rock-vs-hiphop.csv")
echonest_metrics = pd.read_json("datasets/echonest-metrics.json", precise_float=True)

echo_tracks = echonest_metrics.merge(tracks[["track_id", "genre_top"]], on="track_id")
echo_tracks.info()

# create correlation matrix to avoid feature redundancy
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()


features = echo_tracks.drop(["genre_top", "track_id"], axis=1).values

labels = echo_tracks["genre_top"].values

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=10)
