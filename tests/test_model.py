import torch
from src.model import CNN


def test_forward_pass():
    model = CNN()
    model.eval()
    dummy = torch.randn(1, 1, 28, 28)
    out = model(dummy)
    assert out.shape == (1, 10)


def test_batch_forward():
    model = CNN()
    model.eval()
    batch = torch.randn(32, 1, 28, 28)
    out = model(batch)
    assert out.shape == (32, 10)
