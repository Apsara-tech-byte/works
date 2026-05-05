def ex11():
    print(''' 11 - Genome Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample genome dataset
data = {
    "Gene": ["BRCA1", "TP53", "EGFR", "BRCA2", "MYC"],
    "Expression_Level": [5.6, 7.2, 6.1, 5.9, 8.3],
    "Mutation": [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Display dataset
print(df)

# Basic statistics
print("\nStatistics:\n", df.describe())

# Visualization
sns.barplot(x="Gene", y="Expression_Level", data=df)
plt.title("Gene Expression Levels")
plt.show()

# Correlation
correlation = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", correlation)

sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.show()
''')
    