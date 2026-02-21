import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import json
import numpy as np

from .config import train, DEVICE
from .data_loader import PROJECT_ROOT, get_mnist_loaders
from .model import CNN

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Optimize CPU usage
torch.set_num_threads(os.cpu_count())

os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
os.makedirs(PROJECT_ROOT / "results", exist_ok=True)

train_loader, test_loader = get_mnist_loaders()
model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=train['l_r'], weight_decay=train['weight_decay'])

print(f"Device: {DEVICE} | Total batches per epoch: {len(train_loader)}")

# Tracking metrics
train_losses = []
val_losses = []
metrics = {
    'epochs': train['epochs'],
    'learning_rate': train['l_r'],
    'batch_size': train_loader.batch_size,
    'seed': SEED
}

for epoch in range(train['epochs']):
  running_loss = 0.0
  epoch_start = time.time()
  
  # Training phase
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(DEVICE), target.to(DEVICE)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    
    # Print progress every 50 batches
    if (batch_idx + 1) % 100 == 0:
      print(f"Epoch {epoch+1} [{batch_idx + 1}/{len(train_loader)}] - Batch Loss: {loss.item():.4f}")

  avg_train_loss = running_loss / len(train_loader)
  train_losses.append(avg_train_loss)
  
  # Validation phase
  model.eval()
  val_loss = 0.0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(DEVICE), target.to(DEVICE)
      output = model(data)
      loss = criterion(output, target)
      val_loss += loss.item()
  
  avg_val_loss = val_loss / len(test_loader)
  val_losses.append(avg_val_loss)
  
  epoch_time = time.time() - epoch_start
  print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.2f}s")


# Evaluation on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
  for data, target in test_loader:
    data, target = data.to(DEVICE), target.to(DEVICE)
    output = model(data)
    _, predicted = torch.max(output.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

print("[DONE] Training complete!")

# === SAVE METRICS AND TRAINING HISTORY ===
metrics['test_accuracy'] = test_accuracy
metrics['train_losses'] = train_losses
metrics['val_losses'] = val_losses
metrics_path = PROJECT_ROOT / "results" / "training_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")

# === LOSS GRAPH ===
plt.figure(figsize=(10, 6))
epochs_range = range(1, len(train_losses) + 1)
plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training vs Validation Loss', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_graph_path = PROJECT_ROOT / "results" / "loss_graph.png"
plt.savefig(loss_graph_path, dpi=150)
print(f"Loss graph saved to {loss_graph_path}")
plt.show()  

# === SAVE MODEL ===
model_path = "models/mnist_cnn.pth"
torch.save(model.state_dict(), model_path)
print(f" Model saved to {model_path}")

# === TEST ON SINGLE IMAGE ===
model.eval()
example, true_label = test_loader.dataset[0]

with torch.no_grad():
  example_batch = example.unsqueeze(0).to(DEVICE)
  output = model(example_batch)
  predicted = output.argmax(dim=1).item()

print(f" First test image: True = {true_label}, Predicted = {predicted}")

# === VISUALIZE ===
plt.imshow(example.squeeze(), cmap='gray')
plt.title(f"True: {true_label}, Pred: {predicted}")
plt.savefig(PROJECT_ROOT / "results" / "sec_prediction.png")
print(" Image saved to results/sec_prediction.png")


