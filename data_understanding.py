import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import seaborn as sns

# load data
labels_path = 'training_set_labels_cleaned.csv'
features_path = 'training_set_features_cleaned.csv'
labels = pd.read_csv(labels_path)
features = pd.read_csv(features_path)


# configure plot
plt.figure(figsize=(12, 5))
colors = ["#E74C3C", "#2ECC71"] 

# plot H1N1-vaccine distribution
plt.subplot(1, 2, 1)
ax1 = sns.countplot(x='h1n1_vaccine', data=labels, palette=colors)
plt.title('distribution H1N1 vaccine')
total_h1n1 = len(labels['h1n1_vaccine'])
for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2., height, '{:1.2f}%'.format(100 * height/total_h1n1), ha="center")

# plot seasonal vaccine distribution
plt.subplot(1, 2, 2)
ax2 = sns.countplot(x='seasonal_vaccine', data=labels, palette=colors)
plt.title('distribution seasonal vaccine')
total_seasonal = len(labels['seasonal_vaccine'])
for p in ax2.patches:
    height = p.get_height()
    ax2.text(p.get_x()+p.get_width()/2., height, '{:1.2f}%'.format(100 * height/total_seasonal), ha="center")

plt.show()