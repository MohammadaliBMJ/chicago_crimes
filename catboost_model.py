from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_recall_curve
import time
import os


# Load Data
df = pd.read_csv("chicago_dataset_cleaned.csv")

# Split Data
X = df.drop(columns = ["Arrest"])
y = df["Arrest"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 50)

