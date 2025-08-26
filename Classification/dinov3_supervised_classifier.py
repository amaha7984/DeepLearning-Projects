import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, AutoModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import ptwt
import pywt
import torch.nn.functional as F

# ======================== DWT Module ========================
class HaarTransform(nn.Module):
    def __init__(self, level=1, mode="symmetric", with_grad=False):
        super().__init__()
        self.wavelet = pywt.Wavelet("haar")
        self.level = level
        self.mode = mode
        self.with_grad = with_grad

    def forward(self, x):
        with torch.set_grad_enabled(self.with_grad):
            Yl, *Yh = ptwt.wavedec2(x.float(), wavelet=self.wavelet, level=self.level, mode=self.mode)
            if len(Yh) < 1 or len(Yh[0]) != 3:
                raise ValueError("DWT failed: not enough subbands.")
            xH, xV, xD = Yh[0]
            # print(f"[DWT] Shapes: Yl={Yl.shape}, xH={xH.shape}, xV={xV.shape}, xD={xD.shape}")

            Yl = F.interpolate(Yl, size=(224, 224), mode='bilinear', align_corners=False)
            xH = F.interpolate(xH, size=(224, 224), mode='bilinear', align_corners=False)
            xV = F.interpolate(xV, size=(224, 224), mode='bilinear', align_corners=False)
            xD = F.interpolate(xD, size=(224, 224), mode='bilinear', align_corners=False)
            return Yl, xH, xV, xD

# ======================== DINOv2 with DWT ========================
class DINOv3WithDWT(nn.Module):
    def __init__(self, num_classes=2, use_all_bands=True):
        super().__init__()
        self.dwt = HaarTransform()
        self.use_all_bands = use_all_bands
        # DINOv3 backbone (no classification head)
        REPO_DIR = "/aul/homes/amaha038/DeepLearning/Classification/dinov3"  # folder needs to be cloned
        WEIGHTS  = "/aul/homes/amaha038/DeepLearning/Classification/dinov3/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" 
        self.backbone = torch.hub.load(
            REPO_DIR, 'dinov3_vitb16', source='local', weights=WEIGHTS
        ).eval()

        # hidden size of embeddings
        in_features = self.backbone.config.hidden_size

        # classification head
        self.classifier = nn.Linear(in_features * (4 if use_all_bands else 1), num_classes)

    def forward(self, x):

        Yl, xH, xV, xD = self.dwt(x)

        feat_Yl = self.backbone(pixel_values=Yl).pooler_output
        print(f"[BACKBONE] Yl out: {feat_Yl.shape}")

        if self.use_all_bands:
            feat_xH = self.backbone(pixel_values=xH).pooler_output
            feat_xV = self.backbone(pixel_values=xV).pooler_output
            feat_xD = self.backbone(pixel_values=xD).pooler_output
            # print(f"[BACKBONE] xH={feat_xH.shape}, xV={feat_xV.shape}, xD={feat_xD.shape}")
            features = torch.cat([feat_Yl, feat_xH, feat_xV, feat_xD], dim=1)
            print(f"[CONCAT] Combined feature shape: {features.shape}")
        else:
            features = feat_Yl

        logits = self.classifier(features)
        if not hasattr(self, "_dbg_once"):
            print(f"[DEBUG] Logits: {logits.shape}")  # expect (B, {num_classes})
            self._dbg_once = True
        return logits

# ======================== Config ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path = "/aul/homes/amaha038/DetectionGeneratedImages/rine/data/train/Satellite_Combined"
val_path = "/aul/homes/amaha038/DetectionGeneratedImages/rine/data/val/Satellite_Combined"
save_path = "./weights"
os.makedirs(save_path, exist_ok=True)

num_classes = 2
num_epochs = 100
batch_size = 64
learning_rate = 5e-6
weight_decay = 0.05

# ======================== Transforms ========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# ======================== DataLoaders ========================
train_dataset = ImageFolder(train_path, transform=transform)
val_dataset = ImageFolder(val_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ======================== Model ========================
model = DINOv3WithDWT(num_classes=num_classes, use_all_bands=True).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = nn.CrossEntropyLoss()

# ======================== Training ========================
train_loss_log, val_loss_log = [], []
train_acc_log, val_acc_log = [], []
best_val_acc = 0.0
best_val_loss = float("inf")
early_stop_counter = 0
early_stop_patience = 20

print("Starting fine-tuning with DWT...")

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x_batch, y_batch in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total * 100
    train_loss_log.append(avg_train_loss)
    train_acc_log.append(train_accuracy)

    # ======================== Validation ========================
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total * 100
    val_loss_log.append(avg_val_loss)
    val_acc_log.append(val_accuracy)

    scheduler.step()

    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
          f"| Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

    # Save best models
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), os.path.join(save_path, "best_val_acc_pretrained_dinov3_dwt_64batch_final.pth"))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(save_path, "best_val_loss_pretrained_dinov3_dwt_64batch_final.pth"))
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}: No improvement for {early_stop_patience} epochs.")
        break
