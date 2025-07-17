import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*54*54, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

MODEL_REGISTRY = {
    'simple_cnn': SimpleCNNClassifier,
}

def get_model(name, num_classes):
    assert name in MODEL_REGISTRY, f"Model '{name}' not found."
    return MODEL_REGISTRY[name](num_classes)