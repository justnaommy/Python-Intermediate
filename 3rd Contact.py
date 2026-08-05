# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# 2. Load Dataset
train_df = pd.read_csv("Titanic.csv")
test_df = pd.read_csv("Titanic 2.csv")


# Fill missing values
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
test_df["Age"] = test_df["Age"].fillna(train_df["Age"].median())
test_df["Fare"] = test_df["Fare"].fillna(train_df["Fare"].median())
test_df["Embarked"] = test_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

# Encode categorical data
le_sex = LabelEncoder()
combined_sex = pd.concat([train_df["Sex"], test_df["Sex"]])
le_sex.fit(combined_sex)
train_df["Sex"] = le_sex.transform(train_df["Sex"])
test_df["Sex"] = le_sex.transform(test_df["Sex"])

le_embarked = LabelEncoder()
combined_embarked = pd.concat([train_df["Embarked"], test_df["Embarked"]])
le_embarked.fit(combined_embarked)
train_df["Embarked"] = le_embarked.transform(train_df["Embarked"])
test_df["Embarked"] = le_embarked.transform(test_df["Embarked"])

# Create new feature: FamilySize
train_df["FamilySize"] = train_df["Sibsp"] + train_df["Parch"] + 1

features = [ 'Sex', 'Age', 'Sibsp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
X = train_df[features]
y = train_df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))



# Fill missing values
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
test_df["Age"] = test_df["Age"].fillna(train_df["Age"].median())
test_df["Fare"] = test_df["Fare"].fillna(train_df["Fare"].median())
test_df["Embarked"] = test_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

# Convert to lowercase string and strip spaces for safety
train_df["Sex"] = train_df["Sex"].astype(str).str.lower().str.strip()
test_df["Sex"] = test_df["Sex"].astype(str).str.lower().str.strip()

# Combine for consistent LabelEncoder training
combined_sex = pd.concat([train_df["Sex"], test_df["Sex"]])

# Then encode
le = LabelEncoder()
le.fit(combined_sex)

train_df["Sex"] = le.transform(train_df["Sex"])
test_df["Sex"] = le.transform(test_df["Sex"])

# Combine both columns for consistent encoding
combined_sex = pd.concat([train_df["Sex"], test_df["Sex"]])

# Fit encoder on combined data
le = LabelEncoder()
le.fit(combined_sex)

# Now transform both separately
train_df["Sex"] = le.transform(train_df["Sex"])
test_df["Sex"] = le.transform(test_df["Sex"])


# Create FamilySize
test_df["FamilySize"] = test_df["Sibsp"] + test_df["Parch"] + 1

# Select features
X_test = test_df[features]

# Predict
test_predictions = model.predict(X_test)
test_df["PassengerId"] = np.arange(892, 892 + len(test_df))
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv("titanic_submission.csv", index=False)
print("✅ Submission saved as titanic_submission.csv")


