# supervised learning and feature selection to pick the most important feature that could help predict the best crop for the field

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

crops = pd.read_csv("soil_measures.csv")
crops.isna().sum()
crops["crop"].unique()

X = crops[['N', 'P', 'K', 'ph']]
y = crops["crop"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(max_iter=2000, multi_class="multinomial")
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    feature_performance = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-score for {feature}: {feature_performance}")

# F1-score for N: 0.10516656669570501
# F1-score for P: 0.12194454350193666
# F1-score for K: 0.21580486168010007
# F1-score for ph: 0.06787631271947597
  
# the best predictive feature is "K"
