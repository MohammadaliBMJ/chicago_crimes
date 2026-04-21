import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import time
import os
from mlp_model import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Read data
df = pd.read_csv("../data/chicago_dataset_cleaned.csv")
X = df.drop(columns = ['Arrest'])
y = df['Arrest'].astype("float32")


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify = y, random_state = 50)
# Eval data
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, train_size = 0.8, stratify = y_train, random_state = 50)

# Get numerical columns
columns = X.select_dtypes(include = ["int32", "float32", "int64", "float64"]).columns

# Scale X_train and X_test. Skip bool features
scaler = StandardScaler().fit(X_train[columns])

X_train.loc[:, columns] = scaler.transform(X_train[columns])
X_test.loc[:, columns] = scaler.transform(X_test[columns])
X_eval.loc[:, columns] = scaler.transform(X_eval[columns])

# Dummy classifier for comparison
dummy = DummyClassifier(strategy = "most_frequent", random_state = 50)
dummy.fit(X_train, y_train)
dummy_predicts = dummy.predict(X_test)

# Dummy metrics
precision_dummy = precision_score(y_true = y_test, y_pred = dummy_predicts, zero_division = 0)
accuracy_dummy = accuracy_score(y_true = y_test, y_pred = dummy_predicts)
recall_dummy = recall_score(y_true = y_test, y_pred = dummy_predicts)
f1_dummy = f1_score(y_true = y_test, y_pred = dummy_predicts)

X_train["Domestic"] = X_train["Domestic"].astype("int64")
X_test["Domestic"] = X_test["Domestic"].astype("int64")
X_eval["Domestic"] = X_eval["Domestic"].astype("int64")

# Convert to tensor
X_train = torch.tensor(X_train.values, dtype = torch.float32)
X_test = torch.tensor(X_test.values, dtype = torch.float32)
y_train = torch.tensor(y_train.values, dtype = torch.float32)
y_test = torch.tensor(y_test.values, dtype = torch.float32)
X_eval = torch.tensor(X_eval.values, dtype = torch.float32)
y_eval = torch.tensor(y_eval.values, dtype = torch.float32)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_eval = X_eval.to(device)
y_eval = y_eval.to(device)
X_test = X_test.to(device)

# Data Loader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
eval_data = TensorDataset(X_eval, y_eval)

train_loader = DataLoader(
    train_data,
    batch_size = 524288,
    shuffle = True,
    num_workers = 0,
    pin_memory = False
)

test_loader = DataLoader(
    test_data,
    batch_size = 524288,
    shuffle = False,
    num_workers = 0,
    pin_memory = False
)

eval_loader = DataLoader(
    eval_data,
    batch_size = 524288,
    shuffle = False,
    num_workers = 0,
    pin_memory = False
)


# Positive weight
pos_weight = torch.tensor([((y_train == 0).sum()) / ((y_train == 1).sum())], 
                          dtype = torch.float32, device = device)

# Model
model = MLPClassifier(0.2)
# Move model to gpu
model = model.to(device)

loss_function = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5)


# Train. 30 epochs
print("Initializing Training the model...")
start_time = time.time()

for i in range(30):
    model.train()
    start_epoch_time = time.time()
    batch_loss = 0
    eval_epoch_loss = 0

    for x, y in train_loader:

        optimizer.zero_grad()
        predict = model(x)
        loss = loss_function(predict, y.unsqueeze(1))

        loss.backward()
        optimizer.step()

        batch_loss += loss.item()

    with torch.no_grad():
        model.eval()
        for x_eval, y_eval_batch in eval_loader:

            eval_predict = model(x_eval)
            eval_loss = loss_function(eval_predict, y_eval_batch.unsqueeze(1))

            eval_epoch_loss += eval_loss.item()            


    scheduler.step()

    end_epoch_time = time.time()
    epoch_time = end_epoch_time - start_epoch_time
    print(f"Epoch {i + 1}. Loss: {batch_loss / len(train_loader)}. Time to train: {epoch_time:.2f}")
    print(f"Evaluation Loss: {eval_epoch_loss / len(eval_loader)}")

end_time = time.time()
train_time = end_time - start_time
print(f"Training is done. final loss: {batch_loss / len(train_loader)}. Time to train: {train_time}\n")

# Inference
print("Starting inference on test data.")
start_time = time.time()
predictions = []

model.eval()
with torch.no_grad():
    for x, _ in test_loader:
        predict = model(x)
        predict = torch.sigmoid(predict).detach()
        predictions.extend((predict > 0.5).cpu().numpy().astype(int).tolist())

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference is finished. Time: {inference_time:.2f} seconds")

# Model metrics
y_test = y_test.cpu().numpy()
accuracy = accuracy_score(y_true = y_test, y_pred = predictions)
precision = precision_score(y_true = y_test, y_pred = predictions)
recall = recall_score(y_true = y_test, y_pred = predictions)
f1 = f1_score(y_true = y_test, y_pred = predictions)

# Metrics dataframe
metrics_df = pd.DataFrame({
    "model": ["MLP", "Dummy"],
    "Precision": [precision, precision_dummy],
    "Accuracy": [accuracy, accuracy_dummy],
    "Recall": [recall, recall_dummy],
    "F1": [f1, f1_dummy],
    "Train Time": [train_time, 0],
    "Inference Time": [inference_time, 0]
})

# Save results
os.makedirs("../results", exist_ok = True)
metrics_df.to_csv("../results/mlp_results.csv", index = False)
