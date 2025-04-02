import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import xgboost as xgb 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

data = pd.read_csv("Data/FID/good.csv")
print(data.head(6))

x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['ILI_CASE']), data["ILI_CASE"], test_size=.2)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# Create and train XGBoost model with memory-efficient parameters
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=50,  # Reduced from 100
    learning_rate=0.1,
    max_depth=3,      # Reduced from 5
    subsample=0.8,    # Use only 80% of samples per tree
    colsample_bytree=0.8,  # Use only 80% of features per tree
    tree_method='hist',    # Use histogram-based algorithm (more memory efficient)
    random_state=42
)

# Train the model with verbose output to monitor progress
xgb_model.fit(
    x_train, 
    y_train,
    verbose=True
)

# Make predictions
y_pred = xgb_model.predict(x_test)
y_pred_proba = xgb_model.predict_proba(x_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model)
plt.show()
