import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import train, DEVICE
from .data_loader import PROJECT_ROOT, get_mnist_loaders
from .model import SimpleNN    

os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
os.makedirs(PROJECT_ROOT / "results", exist_ok=True)

train_loader, test_loader = get_mnist_loaders()
model = SimpleNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=train['l_r'], weight_decay=train['weight_decay'])

for epoch in range(train['epochs']):
  running_loss = 0.0
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(DEVICE), target.to(DEVICE)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(f"Epoch {epoch+1}: AVG_Loss = {running_loss/len(train_loader):.4f}")


# Evaluation 
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

print(f"Test Accuracy: {100 * correct / total:.2f}%")


print("[DONE] Training complete!")  

# === SAVE MODEL ===
model_path = "models/mnist_model.pth"
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
plt.savefig(PROJECT_ROOT / "results" / "first_prediction.png")
print(" Image saved to results/first_prediction.png")


