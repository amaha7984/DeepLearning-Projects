import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops.layers.torch import Rearrange
from einops import repeat
from torch import Tensor

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

#####################################################
#Below is the implementation of ViTfromScratch Class

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=dim,
                                         num_heads=n_heads,
                                         dropout=dropout,
                                         batch_first=True) 

    def forward(self, x):
        # x: [batch, seq_len, dim] because batch_first=True
        attn_output, _ = self.att(x, x, x)  # Q=K=V=x
        return attn_output

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
        
class ViTfromScratch(nn.Module):
    def __init__(self, ch=3, img_size=224, patch_size=4, emb_dim=32,
                n_layers=6, out_dim=53, dropout=0.1, heads=2): #out_dim=53, matching the number of classes
        super(ViTfromScratch, self).__init__()

        # Attributes
        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                              patch_size=patch_size,
                                              emb_size=emb_dim)
        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout = dropout))))
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))


    def forward(self, img):
        # Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        return self.head(x[:, 0, :])

#End of ViTfromScratch Class
#####################################################

class Mobilev3Classifier(nn.Module):
    def __init__(self, num_classes=53):
        super(Mobilev3Classifier, self).__init__()
        self.base_model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])  # Removing classifier

        feature_out = 1280  # This is MobileNetV3's last feature dimension
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_out, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


MODEL_REGISTRY = {
    'simple_cnn': SimpleCNNClassifier,
    'mobilenetv3': Mobilev3Classifier,
    'vit_scratch': ViTfromScratch,
}


def get_model(name, num_classes):
    assert name in MODEL_REGISTRY, f"Model '{name}' not found."
    return MODEL_REGISTRY[name](num_classes)