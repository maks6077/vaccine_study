import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# load data
features_path = 'training_set_features_encoded.csv'
labels_path = 'training_set_labels_cleaned.csv'
test_features_path = 'test_set_features_encoded.csv'
submission_format_path = 'submission_format.csv'
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)
test_features = pd.read_csv(test_features_path)
submission_format = pd.read_csv(submission_format_path)


# clean column names as model has problems with those symbols
def clean_column_names(df):
    df.columns = [col.replace(">", "gt").replace("<", "lt").replace("[", "").replace("]", "").replace(",", "").replace("=", "eq") for col in df.columns]
    return df


# data preperation
X = features.drop(['respondent_id'], axis=1)
y_h1n1 = labels['h1n1_vaccine']
y_seasonal = labels['seasonal_vaccine']
X_test_submission = test_features.drop(['respondent_id'], axis=1)
X = clean_column_names(X)
X_test_submission = clean_column_names(X_test_submission)

# hyper parameters
params_h1n1 = {'n_estimators': 837, 'learning_rate': 0.059377378770111296, 'max_depth': 3, 'subsample': 0.9448834828833715, 'colsample_bytree': 0.8180301609939452}
params_seasonal = {'n_estimators': 429, 'learning_rate': 0.02729221792750674, 'max_depth': 9, 'subsample': 0.8743368970026256, 'colsample_bytree': 0.698013566864418}

# initialize and train models
model_h1n1 = xgb.XGBClassifier(**params_h1n1, use_label_encoder=False, eval_metric='logloss')
model_seasonal = xgb.XGBClassifier(**params_seasonal, use_label_encoder=False, eval_metric='logloss')
model_h1n1.fit(X, y_h1n1)
model_seasonal.fit(X, y_seasonal)

# predictions for the testset
h1n1_proba = model_h1n1.predict_proba(X_test_submission)[:, 1]
seasonal_proba = model_seasonal.predict_proba(X_test_submission)[:, 1]


### plot feature importance for both models
fig, axs = plt.subplots(2, 1, figsize=(20, 6))

# H1N1
xgb.plot_importance(model_h1n1, max_num_features=10, importance_type='weight', height=0.5, ax=axs[0], show_values=False)
axs[0].set_title('XGBoost: Feature Importance for H1N1 Vaccine')
axs[0].grid(False) 
# seasonal
xgb.plot_importance(model_seasonal, max_num_features=10, importance_type='weight', height=0.5, ax=axs[1], show_values=False)
axs[1].set_title('XGBoost: Feature Importance for Seasonal Flu Vaccine')
axs[1].grid(False) 

plt.tight_layout()
plt.show()


# create submission file
submission = submission_format.copy()
submission['h1n1_vaccine'] = h1n1_proba
submission['seasonal_vaccine'] = seasonal_proba

submission.to_csv('submission_xgb.csv', index=False)

print("submission saved")