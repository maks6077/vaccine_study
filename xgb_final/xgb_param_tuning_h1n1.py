import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
features_path = 'training_set_features_encoded.csv'
labels_path = 'training_set_labels.csv'
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# merge features and labels and clean special characters in columns
data = features.merge(labels, on='respondent_id')
data.columns = [col.replace(">", "gt").replace("<", "lt").replace("[", "").replace("]", "") for col in data.columns]

# split data into features (X) and labels (y) for H1N1
X = data.drop(['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine'], axis=1)
y_h1n1 = data['h1n1_vaccine']

# split into training and test set
X_train, X_test, y_train_h1n1, y_test_h1n1 = train_test_split(X, y_h1n1, test_size=0.2, random_state=42)

# hyperparameter tuning
def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    clf = xgb.XGBClassifier(**param)
    clf.fit(X_train, y_train_h1n1, eval_set=[(X_test, y_test_h1n1)], early_stopping_rounds=50, verbose=True)
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test_h1n1, preds)
    return accuracy

# create optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)


# show top 10 studies (parameter combinations)
top10_trials = sorted(study.trials, key=lambda trial: trial.value, reverse=True)[:10]
for i, trial in enumerate(top10_trials, start=1):
    print(f"Rank {i}: Score {trial.value}, Params {trial.params}")