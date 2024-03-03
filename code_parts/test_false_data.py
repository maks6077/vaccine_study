import pandas as pd
import numpy as np

# load data
file_path = "C:\\Users\\maxim\\Desktop\\Studienarbeit\\training_set_features_cleaned.csv"
data = pd.read_csv(file_path)


# check for respondent_id's with 20 or more missing values
data['missing_values_count'] = data.isnull().sum(axis=1)
ids_to_remove = data[data['missing_values_count'] >= 20]['respondent_id']
print("IDs zum Entfernen:", ids_to_remove.tolist())

# list of categorical columns
categorical_cols = ['employment_occupation', 'employment_industry', 'health_insurance', "opinion_h1n1_vacc_effective",
                   "opinion_h1n1_risk", "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc",
                    'income_poverty', 'rent_or_own', 'employment_status', 
                    'marital_status', 'education', 'doctor_recc_h1n1', 
                    'doctor_recc_seasonal', 'chronic_med_condition', 
                    'child_under_6_months', 'health_worker']  

# check for unique values
for col in categorical_cols:
    print(f"Einzigartige Werte in {col}: {data[col].unique()}\n")

