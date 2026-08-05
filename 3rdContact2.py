
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("spam_ham_dataset.csv", encoding='latin-1')

# Select relevant columns
df = df[['label', 'text', 'label_num']]

# Features and label
X = df['text']
y = df['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📝 Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Test on first 5 samples
sample_texts = df.loc[0:4, 'text']
actual_labels = df.loc[0:4, 'label_num']
predicted_labels = pipeline.predict(sample_texts)

# Show actual vs predicted
for i in range(5):
    print(f"Index: {i} | Actual: {actual_labels.iloc[i]} ({df.loc[i, 'label']})"
          f" | Predicted: {predicted_labels[i]} ({'spam' if predicted_labels[i]==1 else 'ham'})")
