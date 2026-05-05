def ex7():
    print('''
Classification Models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the Pima Indians Diabetes Dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)

# Prepare data for modeling
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
log_accuracy = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {log_accuracy:.4f}")

# Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

# Example prediction for a new patient
new_patient = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  # Example values
new_patient_scaled = scaler.transform(new_patient)

# Get predictions
log_prob = log_reg.predict_proba(new_patient_scaled)[0][1]
nb_prob = nb_model.predict_proba(new_patient_scaled)[0][1]

print(f"\nNew Patient Diabetes Probability:")
print(f"Logistic Regression: {log_prob:.4f}")
print(f"Naive Bayes: {nb_prob:.4f}")
    
    ''')