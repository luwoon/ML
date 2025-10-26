# Train classifier to distinguish between Hip-Hop and Rock  
# Predict whether a song's genre can be correctly classified based on features such as danceability, energy, acousticness, and tempo, etc. 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

tracks = pd.read_csv("datasets/fma-rock-vs-hiphop.csv")
echonest_metrics = pd.read_json("datasets/echonest-metrics.json", precise_float=True)
# Merge relevant columns
echo_tracks = echonest_metrics.merge(tracks[["track_id", "genre_top"]], on="track_id")
echo_tracks.info()

# Create correlation matrix to avoid feature redundancy
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()

features = echo_tracks.drop(["genre_top", "track_id"], axis=1).values
labels = echo_tracks["genre_top"].values

# Split data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=10)

# Normalize feature data
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

# Get explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_ 

# Plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')

# Cumulative explained variance plot
cum_exp_variance = np.cumsum(exp_variance)
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.85, linestyle='--')
# Plot shows that 6 features can explain 85% of the variance

pca = PCA(n_components=6, random_state=10)
train_pca = pca.fit_transform(scaled_train_features)
test_pca = pca.transform(scaled_test_features)

# Use lower dimensional PCA projection of the data to classify songs into genres in a decision tree
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca, train_labels)
pred_labels_tree = tree.predict(test_pca)

# Train logistic regression and predict labels for test set
logreg = LogisticRegression(random_state=10)
logreg.fit(train_pca, train_labels)
pred_labels_logit = logreg.predict(test_pca)

# Create classification report for both models
from sklearn.metrics import classification_report
class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)

print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", class_rep_log)

# Subset only hip-hop tracks, and rock tracks
hop_only = echo_tracks.loc[echo_tracks["genre_top"]=="Hip-Hop"]
rock_only = echo_tracks.loc[echo_tracks["genre_top"]=="Rock"]

# Balance data by sampling rocks songs to be the same number as there are hip-hop songs
rock_only = rock_only.sample(hop_only.shape[0], random_state=10)

# Concatenate the dataframes rock_only and hop_only
rock_hop_bal = pd.concat([rock_only, hop_only])

features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1) 
labels = rock_hop_bal['genre_top']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=10)

train_pca = pca.fit_transform(scaler.fit_transform(train_features))
test_pca = pca.transform(scaler.transform(test_features))

# Train decision tree on the balanced data
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca, train_labels)
pred_labels_tree = tree.predict(test_pca)

# Train logistic regression on the balanced data
logreg = LogisticRegression(random_state=10)
logreg.fit(train_pca, train_labels)
pred_labels_logit = logreg.predict(test_pca)

# Compare models
print("Decision Tree: \n", classification_report(test_labels, pred_labels_tree))
print("Logistic Regression: \n", classification_report(test_labels, pred_labels_logit))

# Evaluate models using cross-validation
tree_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                      ("tree", DecisionTreeClassifier(random_state=10))])
logreg_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                        ("logreg", LogisticRegression(random_state=10))])

kf = KFold(n_splits=10)

tree_score = cross_val_score(tree_pipe, features, labels, cv=kf)
logit_score = cross_val_score(logreg_pipe, features, labels, cv=kf)

print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))
