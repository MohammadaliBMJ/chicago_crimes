from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
import time
import os


# Load Data
df = pd.read_csv("../data/chicago_dataset_cleaned.csv")

# Split Data
X = df.drop(columns = ["Arrest"])
y = df["Arrest"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, train_size = 0.8, random_state = 50)

# Calculate class weight
class_weight = compute_class_weight(class_weight = "balanced", classes = np.array([0, 1]), y = y)

# Range of Hyperparameters for Random Search
params_set = {
    "model__depth": np.arange(4, 12),
    "model__learning_rate": np.linspace(0.0001, 0.1, 50),
    "model__iterations": np.arange(256, 512),
    "model__border_count": np.array([16, 32, 64, 128, 256])
}

# Model Pipeline
pipeline = Pipeline([("model", CatBoostClassifier(verbose = 1, 
                                                  random_seed = 50, task_type = "GPU", 
                                                  loss_function = "Logloss"))])

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

print("Initializing random search...\n")
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()
random_search_time = end_time - start_time
print(f"Random Search is Over. Duration: {random_search_time:.2f} Seconds\n\n")

# Get the best HyperParameters
best_hyperparams = {}
for name, value in random_search.best_params_.items():
    name = name.replace("model__", "")
    best_hyperparams[name] = value

# Build pipeline again
pipeline = Pipeline([("model", 
                      CatBoostClassifier(class_weights = class_weight.tolist(), random_seed = 50, 
                                         **best_hyperparams, loss_function = "Logloss", verbose = 1, task_type = "GPU"))])

# Measure train time
print("Training the model...\n")
start_time = time.time()
pipeline.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time
print(f"Training is over. Time: {train_time:.2f} Seconds\n\n")

# Inference
print("Inference...")
start_time = time.time()
predictions = pipeline.predict(X_test)
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference is over. Time {inference_time:.2f}")

# Dummy Classifier
dummy = DummyClassifier(strategy = "most_frequent", random_state = 50)
dummy.fit(X_train, y_train)
dummy_predicts = dummy.predict(X_test)

# Dummy metrics
accuracy_dummy = accuracy_score(y_true = y_test, y_pred = dummy_predicts)
recall_dummy = recall_score(y_true = y_test, y_pred = dummy_predicts)
f1_dummy = f1_score(y_true = y_test, y_pred = dummy_predicts)

# Catboost metrics
precision_catboost = precision_score(y_pred = predictions, y_true = y_test)
accuracy_catboost = accuracy_score(y_test, predictions)
recall_catboost = recall_score(y_test, predictions)
f1_catboost = f1_score(y_test, predictions)

metrics_df = pd.DataFrame({
    "Precision": [precision_catboost, 0],
    "Accuracy": [accuracy_catboost, accuracy_dummy],
    "Recall": [recall_catboost, recall_dummy],
    "F1": [f1_catboost, f1_dummy],
    "Random Search Time": [random_search_time, 0] ,
    "Train Time": [train_time, 0],
    "Inference Time": [inference_time, 0]
})

# Save
os.makedirs("../results", exist_ok = True)
metrics_df.to_csv("../results/catboost_results.csv", index = False)

# Feature Importance Plot
feature_importance = pipeline.named_steps["model"].feature_importances_
features = X.columns
plt.barh(features, feature_importance)
plt.title("Feature Importance CatBoost")
plt.savefig("../results/catboost_feature_importance.png", bbox_inches = "tight")




