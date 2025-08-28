import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
# from transformers import AutoImageProcessor, AutoModel
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

# REPO_DIR = "/aul/homes/amaha038/DeepLearning/Classification/dinov3"  # folder needs to be cloned
# WEIGHTS  = "/aul/homes/amaha038/DeepLearning/Classification/dinov3/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" 
# ======================== DINOv2 with DWT ========================
class DINOv3WithDWT(nn.Module):
    def __init__(self, num_classes=2, use_all_bands=True,
                 repo_dir="/aul/homes/amaha038/DeepLearning/Classification/dinov3",
                 weights="/aul/homes/amaha038/DeepLearning/Classification/dinov3/dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
                 stats="SAT", freeze_backbone=True):
        super().__init__()
        self.dwt = HaarTransform()
        self.use_all_bands = use_all_bands

        # --- Hub backbone (choose the right entry + weights that match) ---
        # e.g. 'dinov3_vitb16' with a vitb16 checkpoint; 'dinov3_vits16' for vits16, etc.
        self.backbone = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights=weights)
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
            self.backbone.eval()

        # --- normalization that matches the pretraining ---
        if stats.upper() == "SAT":
            self.mean, self.std = (0.496, 0.496, 0.496), (0.244, 0.244, 0.244)
        else:  # LVD (web)
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        # --- feature dim (Hub DINOv3 exposes embed_dim/num_features) ---
        self.feat_dim = getattr(self.backbone, "embed_dim", None) or getattr(self.backbone, "num_features", None)
        assert self.feat_dim is not None, "Cannot infer feature dim; ensure DINOv3 Hub model exposes embed_dim."

        self.classifier = nn.Linear(self.feat_dim * (4 if use_all_bands else 1), num_classes)

    def _norm(self, x):
        mean = x.new_tensor(self.mean)[None,:,None,None]
        std  = x.new_tensor(self.std )[None,:,None,None]
        return (x - mean) / std

    def _embed(self, x):
        # Prefer forward_features → class token
        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]     # [B, feat_dim]
        # Fallback: some hub entries return pooled features directly
        y = self.backbone(x)
        if torch.is_tensor(y) and y.dim() == 2:
            return y
        raise RuntimeError("Unexpected backbone output; forward_features lacks 'x_norm_clstoken' and forward() didn't return [B, D].")

    def forward(self, x):
        Yl, xH, xV, xD = self.dwt(x)
        Yl = self._norm(Yl)
        feat_Yl = self._embed(Yl)
        if self.use_all_bands:
            feats = torch.cat([feat_Yl,
                               self._embed(self._norm(xH)),
                               self._embed(self._norm(xV)),
                               self._embed(self._norm(xD))], dim=1)
        else:
            feats = feat_Yl
        return self.classifier(feats)

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
    transforms.Normalize(mean=(0.496, 0.496, 0.496), std=(0.244, 0.244, 0.244)),
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
        torch.save(model.state_dict(), os.path.join(save_path, "best_val_acc_pretrained_dinov3_dwt_64batch_final_sat.pth"))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(save_path, "best_val_loss_pretrained_dinov3_dwt_64batch_final_sat.pth"))
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}: No improvement for {early_stop_patience} epochs.")
        break
