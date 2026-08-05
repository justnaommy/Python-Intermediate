# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 2. Load the data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 3. Define target and features
X = train_df.drop(columns=["Loan_ID", "Loan_Status"])
y = train_df["Loan_Status"].map({"Y": 1, "N": 0})  # Encode target: Y=1, N=0

X_test_final = test_df.drop(columns=["Loan_ID"])

# 4. Identify column types
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

# 5. Define preprocessors
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# 6. Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 7. Split train into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Fit pipeline on training data
pipeline.fit(X_train, y_train)

# 9. Validate model
y_pred = pipeline.predict(X_val)
print("Accuracy on validation set:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# 10. Predict on actual test.csv data
final_predictions = pipeline.predict(X_test_final)

# 11. Prepare submission
submission = pd.DataFrame({
    "Loan_ID": test_df["Loan_ID"],
    "Loan_Status": ['Y' if pred == 1 else 'N' for pred in final_predictions]
})

submission.to_csv("loan_submission.csv", index=False)
print(" Submission file saved as loan_submission.csv")
