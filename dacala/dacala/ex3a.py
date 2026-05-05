def ex3a():
    print("""
print('Information Extraction from Structured Data')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("\nStep 1: Creating Dataset")
data = {
        "Patient_ID": [1, 2, 3, 4, 5],
        "Age": [34, 45, 23, 50, 40],
        "Gender": ["M", "F", "F", "M", "M"],
        "Diagnosis": ["Diabetes", "Hypertension", "Asthma", "Diabetes", "Hypertension"]
    }

df = pd.DataFrame(data)
print("\nInitial DataFrame:")
print(df.head())

print("\nStep 2: Encoding Gender")
df['Gender'] = df['Gender'].map({"M": 0, "F": 1})

print("\nStep 3: One-Hot Encoding Diagnosis")
df = pd.get_dummies(df, columns=['Diagnosis'])

print("\nStep 4: Dropping Patient_ID")
df = df.drop(["Patient_ID"], axis=1, errors='ignore')
print("\nProcessed DataFrame:")
print(df.head())

print("\nStep 5: Converting to Float for PCA")
X = df.astype(float)

print("\nStep 6: Applying PCA")
pca = PCA(n_components=2)
pca_list = pca.fit_transform(X)

pca_df = pd.DataFrame(pca_list, columns=['PCA1', 'PCA2'])
print("\nPCA Result:")
print(pca_df.head())

print("\nStep 7: Visualizing PCA")
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c='blue')
plt.title("PCA Visualization")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

""")