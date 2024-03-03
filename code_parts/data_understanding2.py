import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

# load data
features_path = 'training_set_features_cleaned.csv'
labels_path = 'training_set_labels_cleaned.csv'
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# merge features and labels
data_combined = pd.merge(features, labels, on='respondent_id')

# list of categorical columns
categorical_cols = ['employment_occupation', 'employment_industry',
                    "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
                    'income_poverty','education', 
                    'doctor_recc_seasonal', 'chronic_med_condition', 
                    'child_under_6_months', 'health_worker', 
                    ]  

# store Chi-square test results
chi2_results = []

for var in categorical_cols:
    contingency_table = pd.crosstab(data_combined[var], data_combined['h1n1_vaccine'])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    chi2_results.append((var, p))


chi2_results.sort(key=lambda x: x[1])


# extract variables and p-values 
variables, p_values = zip(*chi2_results)

# plot p-values
plt.figure(figsize=(10, 8))
plt.barh(variables, [-np.log10(p) for p in p_values])
plt.xlabel('-log10(p-value)', fontsize=18)
plt.ylabel('Variables',fontsize=25)
plt.title('p-values of Chi-squared test for categorical variables', fontsize=22)
plt.tick_params(axis='y', which='major', labelsize=14)
plt.tick_params(axis='x', which='major', labelsize=12)
plt.gca().invert_yaxis() 
plt.show()