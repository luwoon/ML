# train classifier to distinguish between Hip-Hop and Rock  
# predict whether a song's genre can be correctly classified based on features such as danceability, energy, acousticness, and tempo, etc. 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

tracks = pd.read_csv("datasets/fma-rock-vs-hiphop.csv")
echonest_metrics = pd.read_json("datasets/echonest-metrics.json", precise_float=True)
# merge relevant columns
echo_tracks = echonest_metrics.merge(tracks[["track_id", "genre_top"]], on="track_id")
echo_tracks.info()

# create correlation matrix to avoid feature redundancy
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()

features = echo_tracks.drop(["genre_top", "track_id"], axis=1).values
labels = echo_tracks["genre_top"].values

# split data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=10)

# normalize feature data
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

# get explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_ 

# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')

# cumulative explained variance plot
cum_exp_variance = np.cumsum(exp_variance)
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.85, linestyle='--')
# plot shows that 6 features can explain 85% of the variance

pca = PCA(n_components=6, random_state=10)
train_pca = pca.fit_transform(scaled_train_features)
test_pca = pca.transform(scaled_test_features)

# use lower dimensional PCA projection of the data to classify songs into genres in a decision tree
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca, train_labels)
pred_labels_tree = tree.predict(test_pca)
