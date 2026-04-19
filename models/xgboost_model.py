from xgboost import XGBClassifier
from xgboost import plot_importance
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.dummy import DummyClassifier


# Load data
df = pd.read_csv("../data/chicago_dataset_cleaned.csv")
X = df.drop(columns = ['Arrest'])
y = df['Arrest']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, train_size = 0.8, random_state = 50)

# HyperParameters
params_set = {
    "model__eta": np.linspace(0.0001, 0.1, 50),
    "model__max_depth": np.arange(4, 12),
    "model__n_estimators": np.arange(256, 512)
}

# Class weight
class_weight = (y == False).sum() / (y == True).sum()

# Pipeline
pipeline = Pipeline([("model", XGBClassifier(objective = "binary:logistic", 
                                             scale_pos_weight = class_weight, random_state = 50, 
                                             verbosity = 1, device = "cuda"))])

# Random Search
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions = params_set,
    n_iter = 20,
    scoring = "f1",
    n_jobs = 1,
    cv = 3,
    verbose = 1,
    random_state = 50
)

print("Initializing Random Search\n")
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()
random_search_time = end_time - start_time
print(f"Random Search Is Over. Time: {random_search_time:.2f}\n\n")

# Extract best HyperParameters
best_params = {}
for name, value in random_search.best_params_.items():
    name = name.replace("model__", "")
    best_params[name] = value

# Rebuilding the pipeline
pipeline = Pipeline([("model", XGBClassifier(objective = "binary:logistic", 
                                             scale_pos_weight = class_weight, random_state = 50,
                                             verbosity = 1, device = "cuda", **best_params))])

# Retrain to Measure single training time
print("Training The best model...")
start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"Training is finished. Time: {train_time:.2f}\n")

# Inference
print("Initializing Inference on test data...")
start_time = time.time()
predictions = pipeline.predict(X_test)
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference Finished.... Time: {inference_time:.2f}")

# Dummy Classifier
dummy = DummyClassifier(strategy = "most_frequent", random_state = 50)
dummy.fit(X_train, y_train)
dummy_predicts = dummy.predict(X_test)

# Dummy metrics
accuracy_dummy = accuracy_score(y_true = y_test, y_pred = dummy_predicts)
recall_dummy = recall_score(y_true = y_test, y_pred = dummy_predicts)
f1_dummy = f1_score(y_true = y_test, y_pred = dummy_predicts)

# XGBoost metrics
precision_xgb = precision_score(y_pred = predictions, y_true = y_test)
accuracy_xgb = accuracy_score(y_test, predictions)
recall_xgb = recall_score(y_test, predictions)
f1_xgb = f1_score(y_test, predictions)


metrics_df = pd.DataFrame({
    "Precision": [precision_xgb, 0],
    "Accuracy": [accuracy_xgb, accuracy_dummy],
    "Recall": [recall_xgb, recall_dummy],
    "F1": [f1_xgb, f1_dummy],
    "Random Search Time": [random_search_time, 0],
    "Train Time": [train_time, 0],
    "Inference Time": [inference_time, 0]
})

os.makedirs("../results", exist_ok = True)
metrics_df.to_csv("../results/xgboost_results.csv", index = False)

plot_importance(pipeline.named_steps["model"], importance_type = "gain")
plt.title("Feature Importance XGBoost 'Gain'")
plt.savefig("../results/xgboost_feature_importance.png", bbox_inches = "tight")



