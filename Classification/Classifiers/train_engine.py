import torch
from tqdm import tqdm
from utils import compute_accuracy

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        c, t = compute_accuracy(outputs, labels)
        correct += c
        total += t
        running_loss += loss.item() * t

    return running_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            c, t = compute_accuracy(outputs, labels)
            correct += c
            total += t
            running_loss += loss.item() * t

    return running_loss / total, correct / total
