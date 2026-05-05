
def ex6():
    print('''
Information Extraction
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample Dataset (Clinical Text)
clinical_text = [
    "Diagnosis: Diabetes",
    "Patient has Hypertension",
    "Diagnosis: Asthma",
    "The patient is diagnosed with Diabetes",
    "Hypertension is confirmed"
]
labels = ["Diabetes", "Hypertension", "Asthma", "Diabetes", "Hypertension"]

# Rule-Based Extraction
def rule_based_extraction(text):
    match = re.search(r"Diagnosis:\s*([\w\s]+)", text)
    return match.group(1).strip() if match else None

rule_based_results = [rule_based_extraction(text) for text in clinical_text]
print("Rule-Based Extraction Results:", rule_based_results)

# Pattern-Based Extraction
def pattern_based_extraction(text):
    patterns = [r"Diagnosis:\s*([\w\s]+)", r"diagnosed with\s*([\w\s]+)", r"has\s*([\w\s]+)"]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

pattern_based_results = [pattern_based_extraction(text) for text in clinical_text]
print("Pattern-Based Extraction Results:", pattern_based_results)

# Machine Learning-Based Extraction
# Step 1: Convert text to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(clinical_text)

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 3: Train and evaluate models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier()
}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")''')
ex6()