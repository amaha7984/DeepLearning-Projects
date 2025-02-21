import torch
import torchvision
import torchvision.transforms as transforms


training_dataset_path = './traindataset'

# Resizing to 224x224 and convert to tensor
training_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


train_dataset = torchvision.datasets.ImageFolder(root=training_dataset_path, transform=training_transforms)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=False)


def get_mean_and_std(loader):
    mean = torch.zeros(3)  # RGB Images
    std = torch.zeros(3)
    total_images_count = 0

    for images, _ in loader:
        batch_size = images.size(0)  
        images = images.view(batch_size, images.size(1), -1)  

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_size

    # Computing final mean and std
    mean /= total_images_count
    std /= total_images_count

    return mean, std

mean, std = get_mean_and_std(train_loader)
print(f"Dataset Mean: {mean}")
print(f"Dataset Std: {std}")
