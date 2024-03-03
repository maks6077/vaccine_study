import pandas as pd

# load data
file_path = "C:\\Users\\maxim\\Desktop\\Studienarbeit\\training_set_features_cleaned.csv"
file_path2 = "C:\\Users\\maxim\\Desktop\\Studienarbeit\\test_set_features_cleaned.csv"
data = pd.read_csv(file_path)
test_data = pd.read_csv(file_path2)

# list of categorical columns
categorical_cols = ['employment_occupation', 'employment_industry', 'health_insurance', "opinion_h1n1_vacc_effective",
                   "opinion_h1n1_risk", "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc",
                    'income_poverty', 'rent_or_own', 'employment_status', 
                    'marital_status', 'education', 'doctor_recc_h1n1', 
                    'doctor_recc_seasonal', 'chronic_med_condition', 
                    'child_under_6_months', 'health_worker', 'age_group', 'race', 'sex', 
                    'hhs_geo_region', 'census_msa']  

# one-hot-encoding
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)
test_data_encoded = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True, dtype=int)

# save encoded data
data_encoded.to_csv("C:\\Users\\maxim\\Desktop\\Studienarbeit\\training_set_features_encoded.csv", index=False)
test_data_encoded.to_csv("C:\\Users\\maxim\\Desktop\\Studienarbeit\\test_set_features_encoded.csv", index=False)