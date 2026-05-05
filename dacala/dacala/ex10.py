def ex10():
    print('''Outbreak Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample outbreak dataset
data = {
    "Temperature": [30, 32, 35, 28, 26, 33, 31, 29],
    "Humidity": [70, 65, 80, 75, 60, 85, 78, 72],
    "Rainfall": [200, 180, 220, 150, 100, 250, 210, 190],
    "Outbreak": [1, 0, 1, 0, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop("Outbreak", axis=1)
y = df["Outbreak"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualization
sns.pairplot(df, hue="Outbreak")
plt.show()

# Example prediction
new_data = pd.DataFrame({
    "Temperature": [34],
    "Humidity": [82],
    "Rainfall": [230]
})

prediction = model.predict(new_data)
print("\nOutbreak Prediction (1 = Yes, 0 = No):", prediction[0])
''')