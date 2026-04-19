from lightgbm import LGBMClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import os


# Load data. Split into features and label
df = pd.read_csv("chicago_dataset_cleaned.csv")
X = df.drop(columns = ['Arrest'])
y = df['Arrest']

# Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 50)

# Parameters
params_set = {
    "model__num_leaves": np.arange(31, 128),
    "model__learning_rate": np.linspace(0.0001, 0.1, 100),
    "model__n_estimators": np.arange(128, 600),
}

# Model pipeline
pipeline = Pipeline([("model", LGBMClassifier(objective = "binary"))])

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

print("Initializing Random Search...")
# Measure Time
start_time = time.time()
# Fit Random Search
random_search.fit(X_train, y_train)
# Measure Time
end_time = time.time()

print(f"Random Search is finished. Duration: {(end_time - start_time):.2f} Seconds.")

# Best Parameters
best_hyperparams = {}
for name, value in random_search.best_params_.items():
    name = name.replace("model__", "")
    best_hyperparams[name] = value

# Train the model again with nest parameters just to evaluate time
# Rebuild the pipeline with best parameters
pipeline = Pipeline([("model", LGBMClassifier(objective = "binary", **best_hyperparams))])

# Train
print("Training Model Begins...")
start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"Training time: {train_time:.2f} Seconds")

# Inference
print("Making Predictions for test data...")
start_time = time.time()
predictions = pipeline.predict(X_test)
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.2f}")


metrics_df = pd.DataFrame([{
    "Precision": precision_score(y_pred = predictions, y_true = y_test),
    "Accuracy": accuracy_score(y_test, predictions),
    "Recall": recall_score(y_test, predictions),
    "F1": f1_score(y_test, predictions),
    "Train Time": train_time,
    "Inference Time": inference_time
}])

# Save as CSV file
os.makedirs("results", exist_ok = True)
metrics_df.to_csv("./results/lightgbm_results.csv", index = False)

# Feature Importance
lgb.plot_importance(random_search.best_estimator_.named_steps["model"], importance_type = "gain")
plt.savefig("results/lightbgm_feature_importance_gain.png", bbox_inches = "tight")

lgb.plot_importance(random_search.best_estimator_.named_steps["model"], importance_type = "split")
plt.savefig("results/lightbgm_feature_importance_split.png", bbox_inches = "tight")
