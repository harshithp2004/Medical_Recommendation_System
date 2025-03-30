import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load Dataset
dataset = pd.read_csv('datasets/Training.csv')

# Splitting features (X) and target (y)
X = dataset.drop(columns=['prognosis'])
y = dataset['prognosis']

# Encoding prognosis labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=20)

# Dictionary of Models
models = {
    'SVC': SVC(kernel='linear'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'MultinomialNB': MultinomialNB()
}

# Train and Evaluate Each Model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy and Confusion Matrix
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Confusion Matrix:\n{cm}\n")
    print("=" * 50)

# Selecting the Best Model (SVC in this case)
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
svc_accuracy = accuracy_score(y_test, y_pred_svc)

print(f"\nFinal Model: SVC")
print(f"SVC Accuracy: {svc_accuracy:.4f}")

# Save the trained model
joblib.dump(svc, 'models/svc.pkl')
print("SVC model saved as 'svc.pkl'")
