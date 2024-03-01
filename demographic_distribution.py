import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
features_path = 'training_set_features.csv'
features = pd.read_csv(features_path)

# set age group order for plot
age_order = ['65+ Years', '55 - 64 Years', '45 - 54 Years', '35 - 44 Years','18 - 34 Years']

# making sure categorical data is formatted correctly
features['age_group'] = pd.Categorical(features['age_group'], categories=age_order, ordered=True)
features['education'] = features['education'].astype('category')
features['sex'] = features['sex'].astype('category')
features['race'] = features['race'].astype('category')


### Plots

# age group
plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
sns.countplot(y='age_group', data=features, order=age_order)
plt.title('Distribution by Age Group')
plt.xlabel('Number of Respondents')

# sex
plt.subplot(4, 1, 2)
sns.countplot(y='sex', data=features)
plt.title('Distribution by Gender')
plt.xlabel('Number of Respondents')

# education
plt.subplot(4, 1, 3)
sns.countplot(y='education', data=features, order=features['education'].value_counts().index)
plt.title('Distribution by Education Level')
plt.xlabel('Number of Respondents')

# race
plt.subplot(4, 1, 4)
sns.countplot(y='race', data=features, order=features['race'].value_counts().index)
plt.title('Distribution by Race')
plt.xlabel('Number of Respondents')


plt.tight_layout()
plt.show()
