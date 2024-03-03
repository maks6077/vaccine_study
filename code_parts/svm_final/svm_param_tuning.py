import pandas as pd
import optuna
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from optuna.samplers import TPESampler

# load data
features_path = 'training_set_features_encoded.csv'
labels_path = 'training_set_labels_cleaned.csv'
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)
X = features.drop(['respondent_id'], axis=1)
y_h1n1 = labels['h1n1_vaccine']
y_seasonal = labels['seasonal_vaccine']

# hyperparameter tuning
def objective(X, y, trial):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    C = trial.suggest_loguniform('C', 1e-3, 1e3)
    kernel = trial.suggest_categorical('kernel', ['linear'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    
    # SVM model
    clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

# optuna study for H1N1
study_h1n1 = optuna.create_study(direction='maximize', sampler=TPESampler())
study_h1n1.optimize(partial(objective, X, y_h1n1), n_trials=100)

print("H1N1 beste Parameter:", study_h1n1.best_params)
print("H1N1 bester Cross-Validation Score:", study_h1n1.best_value)

# optuna study for seasonal
study_seasonal = optuna.create_study(direction='maximize', sampler=TPESampler())
study_seasonal.optimize(partial(objective, X, y_seasonal), n_trials=100)

print("\nseasonal beste Parameter:", study_seasonal.best_params)
print("seasonal bester Cross-Validation Score:", study_seasonal.best_value)
