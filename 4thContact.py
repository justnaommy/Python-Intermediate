from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

# Display dataset info
print(f"Dataset shape: {X.shape}")
print("\nFeature names:", california.feature_names)
print("\nTarget: Median House Value (in $100,000s)")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"- MSE: {mse:.3f}")
print(f"- RMSE: {rmse:.3f} (≈${rmse*100000:.0f} error)")
print(f"- R²: {r2:.3f}")

# Feature importance (from model inside pipeline)
model = pipeline.named_steps['regressor']
plt.figure(figsize=(10, 5))
pd.Series(model.feature_importances_, index=california.feature_names).sort_values().plot.barh()
plt.title('Feature Importance')
plt.show()

# Sample prediction
sample_idx = 10
sample_data = X_test.iloc[sample_idx:sample_idx+1]
actual_price = y_test[sample_idx]
predicted_price = pipeline.predict(sample_data)[0]

print("\nSample Prediction:")
print(f"Actual Value: ${actual_price*100000:.0f}")
print(f"Predicted Value: ${predicted_price*100000:.0f}")
print(f"Difference: ${(predicted_price-actual_price)*100000:.0f}")
