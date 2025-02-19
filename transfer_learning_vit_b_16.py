import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

from torchvision import datasets, models, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.optim import lr_scheduler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mean and standard deviation for normalization - ImageNet stats
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Defining data transformations
def get_data_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    }

# Loading datasets and creating dataloaders
def get_dataloaders(data_dir, batch_size=4, num_workers=4):
    transforms_dict = get_data_transforms()
    sets = ["train", "val"]

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transforms_dict[x]) for x in sets}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == "train"), num_workers=num_workers) for x in sets}

    dataset_sizes = {x: len(image_datasets[x]) for x in sets}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names

# Data directory path
data_dir = "/aul/homes/amaha038/Mapsgeneration/TerraFlyClassification" #"data/folder_name"

# Load dataloaders
dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)

# Model training function
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}\n" + "-" * 10)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step the scheduler only after the training phase
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


def initialize_model(num_classes=2, pretrained=True):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(device)

# Initializing Steps
model = initialize_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
num_epochs = 100

# Training the model
model = train_model(model, criterion, optimizer, scheduler, num_epochs=100)
