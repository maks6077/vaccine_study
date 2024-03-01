import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# load data
features_path = 'training_set_features_encoded.csv'
labels_path = 'training_set_labels_cleaned.csv'
test_features_path = 'test_set_features_encoded.csv'
submission_format_path = 'submission_format.csv'
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)
test_features = pd.read_csv(test_features_path)
submission_format = pd.read_csv(submission_format_path)

# preparing data
X = features.drop(['respondent_id'], axis=1)
y_h1n1 = labels['h1n1_vaccine']
y_seasonal = labels['seasonal_vaccine']
X_test_submission = test_features.drop(['respondent_id'], axis=1)

# scaling of features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_submission_scaled = scaler.transform(X_test_submission)

# defining hyperparameters
optimal_params = {'C': 0.022770434858826624, 'kernel': 'linear', 'gamma': 'auto'}

# initialise models for H1N1 and seasonal
svm_h1n1 = SVC(**optimal_params, probability=True)
svm_seasonal = SVC(**optimal_params, probability=True)

# train SVM model for H1N1
svm_h1n1.fit(X_scaled, y_h1n1)

# train SVM model for seasonal
svm_seasonal.fit(X_scaled, y_seasonal)

# make predictions for testset
h1n1_proba = svm_h1n1.predict_proba(X_test_submission_scaled)[:, 1]
seasonal_proba = svm_seasonal.predict_proba(X_test_submission_scaled)[:, 1]

# save submission file
submission = submission_format.copy()
submission['h1n1_vaccine'] = h1n1_proba
submission['seasonal_vaccine'] = seasonal_proba
submission.to_csv('svm_submission.csv', index=False)

print("SVM submission erfolgreich gespeichert.")
