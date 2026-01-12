import numpy as np

# =========================================================
# Utilities
# =========================================================
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def _ensure_3d(X: np.ndarray):
    # Accept (S, D) or (B, S, D)
    if X.ndim == 2:
        return X[None, ...], True
    if X.ndim == 3:
        return X, False
    raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")

# =========================================================
# Core Layers (NumPy, forward-only)
# =========================================================
class Layer_Dense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones((1, 1, d_model))
        self.beta  = np.zeros((1, 1, d_model))

    def forward(self, X: np.ndarray) -> np.ndarray:
        X3, squeeze_back = _ensure_3d(X)
        mean = np.mean(X3, axis=-1, keepdims=True)
        var  = np.var(X3, axis=-1, keepdims=True)
        Xhat = (X3 - mean) / np.sqrt(var + self.eps)
        out  = self.gamma * Xhat + self.beta
        self.output = out[0] if squeeze_back else out
        return self.output

class Dropout:
    def __init__(self, drop_prob: float = 0.0, seed=None):
        self.drop_prob = float(drop_prob)
        self.rng = np.random.default_rng(seed)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if (not training) or self.drop_prob <= 0.0:
            self.output = X
            return X
        keep = 1.0 - self.drop_prob
        mask = (self.rng.random(X.shape) < keep)
        out = (X * mask) / keep
        self.output = out
        return out

# =========================================================
# Multi-Head Self-Attention (Encoder-style)
# =========================================================
class MultiHeadSelfAttention:
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, seed=None):
        if d_model % n_heads != 0:
            raise ValueError("n_heads must evenly divide d_model")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.W_q = Layer_Dense(d_model, d_model)
        self.W_k = Layer_Dense(d_model, d_model)
        self.W_v = Layer_Dense(d_model, d_model)
        self.W_o = Layer_Dense(d_model, d_model)

        self.attn_dropout = Dropout(dropout, seed=seed)

    def _split_heads(self, X3: np.ndarray) -> np.ndarray:
        # (B,S,D) -> (B,H,S,Dh)
        B, S, D = X3.shape
        H = self.n_heads
        return X3.reshape(B, S, H, self.d_head).transpose(0, 2, 1, 3)

    def _combine_heads(self, X4: np.ndarray) -> np.ndarray:
        # (B,H,S,Dh) -> (B,S,D)
        B, H, S, Dh = X4.shape
        return X4.transpose(0, 2, 1, 3).reshape(B, S, H * Dh)

    def forward(self, X: np.ndarray, attn_mask=None, training: bool = True) -> np.ndarray:
        """
        X: (S,D) or (B,S,D)
        attn_mask: None or array broadcastable to (B,H,S,S), e.g. (S,S) with 0 allowed and -1e9 blocked.
        """
        X3, squeeze_back = _ensure_3d(X)

        Q = self.W_q.forward(X3)
        K = self.W_k.forward(X3)
        V = self.W_v.forward(X3)

        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        scores = np.matmul(Qh, Kh.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)  # (B,H,S,S)
        if attn_mask is not None:
            scores = scores + attn_mask

        probs = softmax(scores, axis=-1)
        probs = self.attn_dropout.forward(probs, training=training)

        out_h = np.matmul(probs, Vh)        # (B,H,S,Dh)
        out   = self._combine_heads(out_h)  # (B,S,D)
        out   = self.W_o.forward(out)       # (B,S,D)

        self.output = out[0] if squeeze_back else out
        return self.output

# =========================================================
# Feed-Forward Network
# =========================================================
class FeedForward:
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, seed=None, activation: str = "gelu"):
        self.fc1 = Layer_Dense(d_model, d_ff)
        self.fc2 = Layer_Dense(d_ff, d_model)
        self.dropout = Dropout(dropout, seed=seed)
        self.activation = activation.lower()

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, x)
        if self.activation == "gelu":
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * (x**3))))
        raise ValueError("activation must be 'relu' or 'gelu'")

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        X3, squeeze_back = _ensure_3d(X)
        h = self.fc1.forward(X3)
        h = self._act(h)
        h = self.dropout.forward(h, training=training)
        out = self.fc2.forward(h)
        self.output = out[0] if squeeze_back else out
        return self.output

# =========================================================
# Transformer Encoder Layer (Pre-LN)
# =========================================================
class TransformerEncoderLayer:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, seed=None, activation: str = "gelu"):
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        self.mha = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, seed=seed)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout, seed=seed, activation=activation)

        self.drop1 = Dropout(dropout, seed=seed)
        self.drop2 = Dropout(dropout, seed=seed)

    def forward(self, X: np.ndarray, attn_mask=None, training: bool = True) -> np.ndarray:
        X3, squeeze_back = _ensure_3d(X)

        # Pre-LN attention
        h1 = self.ln1.forward(X3)
        a  = self.mha.forward(h1, attn_mask=attn_mask, training=training)
        a  = self.drop1.forward(a, training=training)
        X3 = X3 + a

        # Pre-LN FFN
        h2 = self.ln2.forward(X3)
        f  = self.ffn.forward(h2, training=training)
        f  = self.drop2.forward(f, training=training)
        X3 = X3 + f

        self.output = X3[0] if squeeze_back else X3
        return self.output

class TransformerEncoder:
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, seed=None, activation: str = "gelu"):
        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=dropout, seed=seed, activation=activation)
            for _ in range(n_layers)
        ]
        self.final_ln = LayerNorm(d_model)

    def forward(self, X: np.ndarray, attn_mask=None, training: bool = True) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out, attn_mask=attn_mask, training=training)
        out = self.final_ln.forward(out)
        self.output = out
        return out

# =========================================================
# ViT-specific pieces
# =========================================================
class PatchEmbed:
    def __init__(self, img_size: int, patch_size: int, in_chans: int, d_model: int):
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.d_model = d_model

        self.grid = img_size // patch_size
        self.n_patches = self.grid * self.grid
        self.proj = Layer_Dense(patch_size * patch_size * in_chans, d_model)

    def forward(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = img[None, ...]
            squeeze_back = True
        elif img.ndim == 4:
            squeeze_back = False
        else:
            raise ValueError(f"Expected img shape (H,W,C) or (B,H,W,C), got {img.shape}")

        B, H, W, C = img.shape
        P = self.patch_size
        if (H != self.img_size) or (W != self.img_size) or (C != self.in_chans):
            raise ValueError(f"Expected (H,W,C)=({self.img_size},{self.img_size},{self.in_chans}), got ({H},{W},{C})")

        gh = H // P
        gw = W // P
        patches = img.reshape(B, gh, P, gw, P, C).transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(B, gh * gw, P * P * C)

        tokens = self.proj.forward(patches)  # (B,N,D)
        self.output = tokens[0] if squeeze_back else tokens
        return self.output

class ViT:
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.0,
        seed=None,
        activation: str = "gelu",
    ):
        self.rng = np.random.default_rng(seed)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, d_model)
        n_patches = self.patch_embed.n_patches

        self.cls_token = 0.02 * self.rng.standard_normal((1, 1, d_model))
        self.pos_embed = 0.02 * self.rng.standard_normal((1, 1 + n_patches, d_model))

        self.pos_drop = Dropout(dropout, seed=seed)
        self.encoder = TransformerEncoder(n_layers, d_model, n_heads, d_ff, dropout=dropout, seed=seed, activation=activation)

        self.head = Layer_Dense(d_model, num_classes)

    def forward(self, img: np.ndarray, training: bool = False) -> np.ndarray:
        x = self.patch_embed.forward(img)   # (N,D) or (B,N,D)
        x3, _ = _ensure_3d(x)               # (B,N,D)

        B, N, D = x3.shape
        cls = np.repeat(self.cls_token, repeats=B, axis=0)  # (B,1,D)
        x3 = np.concatenate([cls, x3], axis=1)              # (B,1+N,D)

        x3 = x3 + self.pos_embed[:, : (1 + N), :]
        x3 = self.pos_drop.forward(x3, training=training)

        x3 = self.encoder.forward(x3, attn_mask=None, training=training)  # (B,1+N,D)

        cls_out = x3[:, 0, :]                 # (B,D)
        logits = self.head.forward(cls_out)   # (B,num_classes)

        self.output = logits[0] if (img.ndim == 3) else logits
        return self.output

# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    np.random.seed(0)

    vit = ViT(
        img_size=32,
        patch_size=8,
        in_chans=3,
        num_classes=10,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        seed=0,
    )

    img = np.random.randn(32, 32, 3)
    logits = vit.forward(img, training=False)
    print("Single image logits shape:", logits.shape)

    batch = np.random.randn(4, 32, 32, 3)
    logits_b = vit.forward(batch, training=False)
    print("Batch logits shape:", logits_b.shape)
