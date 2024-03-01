import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# load data
file_path = "C:\\Users\\maxim\\Desktop\\Studienarbeit\\training_set_features.csv"
labels_file_path = "C:\\Users\\maxim\\Desktop\\Studienarbeit\\training_set_labels.csv"
data = pd.read_csv(file_path)
labels = pd.read_csv(labels_file_path)

# calculation of missing values per column
missing_values_abs = data.isnull().sum()
missing_values_percent = (data.isnull().sum() / data.shape[0]) * 100

# summarize missing values in a DataFrame
missing_values_summary = pd.DataFrame({
    'Missing Values': missing_values_abs,
    'Percentage (%)': missing_values_percent
}).sort_values(by='Missing Values', ascending=False)
print(missing_values_summary)


# visualization of missing values
plt.figure(figsize=(15, 8))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('missing values in dataset')
plt.show()

# save missing values in CSV
missing_values_summary_path = 'missing_values_summary_trainingset.csv'
missing_values_summary.to_csv(missing_values_summary_path)

# delete records with 19 or more missing values
data['missing_values_count'] = data.isnull().sum(axis=1)
ids_to_remove = data[data['missing_values_count'] >= 19]['respondent_id']
print("IDs entfernt:", ids_to_remove.tolist())
data_cleaned = data[~data['respondent_id'].isin(ids_to_remove)].copy()
labels_cleaned = labels[~labels['respondent_id'].isin(ids_to_remove)].copy()


# analysis of distribution of missing values
missing_values_distribution = data['missing_values_count'].value_counts().sort_index()
print(missing_values_distribution)
data_cleaned.drop(['missing_values_count'], axis=1, inplace=True)

# replace with "missing" if no value in this column
for col in ['employment_occupation', 'employment_industry', 'health_insurance', "opinion_h1n1_vacc_effective", 'income_poverty', 'rent_or_own', 'employment_status',
             "opinion_h1n1_risk", "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc", 'education']:
    data_cleaned[col].fillna('missing', inplace=True)

# imputation for numerical variables with median
numerical_cols = ['household_adults', 'household_children']
num_imputer = SimpleImputer(strategy='median')
data_cleaned[numerical_cols] = num_imputer.fit_transform(data_cleaned[numerical_cols])

# imputation for the remaining variables with most frequently occurring value
categorical_cols = ['marital_status', 'doctor_recc_h1n1', 
                    'doctor_recc_seasonal', 'chronic_med_condition', "behavioral_avoidance", "behavioral_face_mask", "behavioral_wash_hands",
                    'child_under_6_months', 'health_worker', 'h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds', "behavioral_large_gatherings",
                    "behavioral_outside_home", "behavioral_touch_face"]

cat_imputer = SimpleImputer(strategy='most_frequent')
data_cleaned[categorical_cols] = cat_imputer.fit_transform(data_cleaned[categorical_cols])


# save cleaned data in CSV
data_cleaned.to_csv('training_set_features_cleaned.csv', index=False)
labels_cleaned.to_csv('training_set_labels_cleaned.csv', index=False)
print("Preprocessing abgeschlossen.")