import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# load data
features_path = 'training_set_features_encoded.csv'
labels_path = 'training_set_labels.csv'
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# merge features and labels
data = features.merge(labels, on='respondent_id')

# split data into features (X) and labels (y) for both models
X = data.drop(['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine'], axis=1)
y_h1n1 = data['h1n1_vaccine']
y_seasonal = data['seasonal_vaccine']

# split into training and test set
X_train, X_test, y_train_h1n1, y_test_h1n1 = train_test_split(X, y_h1n1, test_size=0.2, random_state=42)
X_train, X_test, y_train_seasonal, y_test_seasonal = train_test_split(X, y_seasonal, test_size=0.2, random_state=42)

# define hyperparameter grid
param_grid = {
    'n_neighbors': [2, 5, 10, 30, 75, 106, 160],
    'metric': ['euclidean', 'manhattan', 'hamming']
}

# function to show top 10 results
def display_top_10_results(grid_search):
    cv_results = grid_search.cv_results_
    top10_idx = np.argsort(cv_results['mean_test_score'])[-10:]
    print("Top 10 Scores und ihre Parameter:")
    for idx in top10_idx[::-1]:
        print(f"Score: {cv_results['mean_test_score'][idx]:.4f}, Params: {cv_results['params'][idx]}")

# GridSearchCV for h1n1 model
grid_search_h1n1 = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=3)
grid_search_h1n1.fit(X_train, y_train_h1n1)
print("Beste Parameter für H1N1-Modell:", grid_search_h1n1.best_params_)
display_top_10_results(grid_search_h1n1)

# GridSearchCV for seasonal model
grid_search_seasonal = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=3)
grid_search_seasonal.fit(X_train, y_train_seasonal)
print("Beste Parameter für saisonales Modell:", grid_search_seasonal.best_params_)
display_top_10_results(grid_search_seasonal)
