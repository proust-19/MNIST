import torch.nn as nn
import torch.nn.functional as F
from .config import model as m

class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__()
    self.fc1 = nn.Linear(m['input_size'], m['h_size'])
    self.bn1 = nn.BatchNorm1d(m['h_size'])
    self.dropout1 = nn.Dropout(m['dropout_rate'])
    self.fc2 = nn.Linear(m['h_size'], m['h_size2'])
    self.bn2 = nn.BatchNorm1d(m['h_size2'])
    self.dropout2 = nn.Dropout(m['dropout_rate'])
    self.fc3 = nn.Linear(m['h_size2'], m['num_classes'])

  def forward(self, x):
    x = x.view(-1, m['input_size'])
    x = self.fc1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.dropout2(x)
    return self.fc3(x)

class CNN(nn.Module):
  def __init__(self):
    super (CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.pool = nn.MaxPool2d(2, 2)

    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(64)

    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.dropout = nn.Dropout(0.05)
    self.fc2 = nn.Linear(128, m['num_classes'])

  def forward(self, x):
    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = x.view(-1, 64 * 7 * 7)

    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    return self.fc2(x)  
  