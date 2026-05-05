def ex4():
    print(''' Implementing Data Mining on Non Clinical Data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Step 1: Create Dataset
data = {
    "Age": np.random.randint(20, 80, 100),
    "BMI": np.random.randint(18, 35, 100),
    "Smoking": np.random.randint(0, 2, 100),
    "Alchohol_Consumption": np.random.randint(0, 4, 100),
    "Physical_Activity": np.random.randint(0, 4, 100),
    "Chronic_Disease": np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)
print(df.head())


# Step 2: Standardization
features = ['Age','BMI','Smoking','Alchohol_Consumption','Physical_Activity','Chronic_Disease']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])


# Step 3: Pairplot
sns.pairplot(df)


# Step 4: Correlation Heatmap
corr = df.corr()
sns.heatmap(corr, annot=True)


# Step 5: PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[features])

pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
pca_df['Chronic_Disease'] = df['Chronic_Disease']

plt.scatter(
    pca_df['PCA1'],
    pca_df['PCA2'],
    c=pca_df['Chronic_Disease']
)


print(pca_df.head())


# Step 6: KMeans Clustering
kmeans = KMeans(n_clusters=2)

features = ['PCA1', 'PCA2']
pca_df['Clusters'] = kmeans.fit_predict(pca_df[features])

plt.scatter(
    pca_df['PCA1'],
    pca_df['PCA2'],
    c=pca_df['Clusters']
)''')