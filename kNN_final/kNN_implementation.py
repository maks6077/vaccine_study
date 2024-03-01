import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# load data
features_path = 'training_set_features_encoded.csv'
labels_path = 'training_set_labels.csv'
test_features_path = 'test_set_features_encoded.csv'
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)
test_features = pd.read_csv(test_features_path)

# merge features and labels
data = features.merge(labels, on='respondent_id')

# split data into features (X) and labels (y) for both models
X = data.drop(['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine'], axis=1)
y_h1n1 = data['h1n1_vaccine']
y_seasonal = data['seasonal_vaccine']

# drop respondent_id from testset
X_test = test_features.drop(['respondent_id'], axis=1)


# model training for h1n1
knn_h1n1 = KNeighborsClassifier(metric='manhattan', n_neighbors=75)
knn_h1n1.fit(X, y_h1n1)

# model training for seasonal
knn_seasonal = KNeighborsClassifier(metric='hamming', n_neighbors=75)
knn_seasonal.fit(X, y_seasonal)

# make predictions for testset
h1n1_vaccine_pred = knn_h1n1.predict_proba(X_test)[:, 1]
seasonal_vaccine_pred = knn_seasonal.predict_proba(X_test)[:, 1]

# saving predictions in CSV
submission = test_features[['respondent_id']].copy()
submission['h1n1_vaccine'] = h1n1_vaccine_pred
submission['seasonal_vaccine'] = seasonal_vaccine_pred
submission.to_csv('submission_knn.csv', index=False)

print("kNN Predictions erfolgreich gespeichert.")
