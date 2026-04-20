import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time
import os
from mlp_model import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd



# Read data
df = pd.read_csv("../data/chicago_crime.csv")
X = df.drop(columns = ['Arrest'])
y = df['Arrest']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify = y, random_state = 50)

# Get numerical columns
columns = X.select_dtypes(include = ["int32", "float32", "int64", "float64"]).columns

# Scale X_train and X_test. Skip bool features
scaler = StandardScaler().fit(X_train[columns])

X_train[columns] = scaler.transform(X_train[columns])
X_test[columns] = scaler.transform(X_test[columns])

# Convert to tensor
X_train = torch.tensor(X_train, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32)

# Data Loader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_data,
    batch_size = 64,
    shuffle = True,
    num_workers = 8,
    pin_memory = True
)

test_loader = DataLoader(
    test_data,
    batch_size = 64,
    shuffle = False,
    num_workers = 8,
    pin_memory = True
)

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Positive weight
pos_weight = torch.tensor([((y_train == 0).sum()) / ((y_train == 1).sum())], 
                          dtype = torch.float32, device = device)

# Model
model = MLPClassifier(0.2)
loss_function = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5)

# Move model to gpu
model = model.to(device)

# Train. 30 epochs
print("Initializing Training the model...")
start_time = time.time()

for i in range(30):
    model.train()
    start_epoch_time = time.time()

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        predict = model(x)
        loss = loss_function(predict, y.unsqueeze(1))

        loss.backward()
        optimizer.step()
    
    scheduler.step(i)

    end_epoch_time = time.time()
    epoch_time = end_epoch_time - start_epoch_time
    print(f"Epoch {i + 1}. Loss: {loss}. Time to train: {epoch_time:.2f}")

end_time = time.time()
train_time = end_time - start_time
print(f"Training is done. final loss: {loss}. Time to train: {train_time}\n")

# Inference
print("Starting inference on test data.")
start_time = time.time()
predictions = []

model.eval()
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        predict = model(x)
        predict = torch.sigmoid(predict).detach()
        predictions.extend((predict > 0.5).cpu().numpy().astype(int).tolist())

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference is finished. Time: {inference_time:.2f} seconds")

y_test = y_test.tolist()
accuracy = accuracy_score(y_true = y_test, y_pred = predictions)
precision = precision_score(y_true = y_test, y_pred = predictions)
recall = recall_score(y_true = y_test, y_pred = predictions)
f1 = f1_score(y_true = y_test, y_pred = predictions)

# Metrics dataframe
metrics_df = pd.DataFrame({
    "Precision": [precision],
    "Accuracy": [accuracy],
    "Recall": [recall],
    "F1": [f1],
    "Train Time": [train_time],
    "Inference Time": [inference_time]
})

# Save results
os.makedirs("../results", exist_ok = True)
metrics_df.to_csv("../results/mlp_results.csv", index = False)
