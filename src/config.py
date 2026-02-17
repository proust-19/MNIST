import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Data configuration
data = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 0  # Set to 0 for Windows, 2-4 for Linux/Mac
}

# Model configuration
model = {
    'input_size': 784,  # 28x28
    'h_size': 128,
    'h_size2': 64,
    'num_classes': 10,
    'dropout_rate': 0.3
}

# Training configuration
train = {
    'l_r': 0.005,
    'epochs': 40,
    'log_interval': 100,  
    'weight_decay': 1e-4
}