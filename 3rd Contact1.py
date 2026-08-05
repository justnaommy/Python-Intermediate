import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin

# Load data
train_df = pd.read_csv("Titanic.csv")
test_df = pd.read_csv("Titanic 2.csv")

# Custom transformer for FamilySize
class FamilySizeAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['FamilySize'] = X['Sibsp'] + X['Parch'] + 1
        return X

# Features & Target
features = ['Sex', 'Age', 'Sibsp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = train_df[features]
y = train_df[target]
X_test_final = test_df[features]

# Preprocessing for numerical and categorical features
numeric_features = ['Age', 'Sibsp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_features = ['Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine all transformers
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
pipeline = Pipeline(steps=[
    ('family', FamilySizeAdder()),
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_val)
print("✅ Accuracy:", accuracy_score(y_val, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("📝 Classification Report:\n", classification_report(y_val, y_pred))

# Prepare test data
test_df["FamilySize"] = test_df["Sibsp"] + test_df["Parch"] + 1
X_test = test_df[features]

# Predict on test set
test_predictions = pipeline.predict(X_test)

# Save submission
test_df["PassengerId"] = np.arange(892, 892 + len(test_df))
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_predictions
})
submission.to_csv("titanic_submission.csv", index=False)
print("🚢 Submission saved as titanic_submission.csv")
