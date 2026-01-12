# =========================================================
# Transformer Encoder Only
# =========================================================
import numpy as np
def softmax(x: np.ndarray) -> np.ndarray:
    # Numerically stable softmax
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def _ensure_3d(X: np.ndarray):
    """
    Accepts X as (S, D) or (B, S, D).
    Returns (X3, squeeze_back) where squeeze_back=True if original was 2D.
    """
    if X.ndim == 2:
        return X[None, ...], True
    if X.ndim == 3:
        return X, False
    raise ValueError(f"Expected X with 2 or 3 dims, got shape {X.shape}")

# =========================================================
# Core Layers (NumPy, forward-only)
# =========================================================
class Layer_Dense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # inputs: (..., n_inputs)
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones((1, 1, d_model))  # broadcast over (B,S,D)
        self.beta  = np.zeros((1, 1, d_model))

    def forward(self, X: np.ndarray) -> np.ndarray:
        X3, squeeze_back = _ensure_3d(X)
        mean = np.mean(X3, axis=-1, keepdims=True)
        var  = np.var(X3, axis=-1, keepdims=True)
        Xhat = (X3 - mean) / np.sqrt(var + self.eps)
        out = self.gamma * Xhat + self.beta
        self.output = out[0] if squeeze_back else out
        return self.output

class Dropout:
    """
    Optional; set drop_prob=0.0 to disable.
    (Forward-only; 'training' flag controls whether mask is applied.)
    """
    def __init__(self, drop_prob: float = 0.0, seed: int | None = None):
        self.drop_prob = float(drop_prob)
        self.rng = np.random.default_rng(seed)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if (not training) or self.drop_prob <= 0.0:
            self.output = X
            return X
        keep_prob = 1.0 - self.drop_prob
        mask = self.rng.random(X.shape) < keep_prob
        out = (X * mask) / keep_prob
        self.output = out
        return out

# =========================================================
# Multi-Head Self-Attention (Encoder-style)
# =========================================================
class MultiHeadSelfAttention:
    """
    Encoder self-attention: uses Q=K=V=Linear(X) with learnable projections.
    Supports X: (S, D) or (B, S, D).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, seed: int | None = None):
        if d_model % n_heads != 0:
            raise ValueError("n_heads must evenly divide d_model")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # Using Dense-style linear layers
        self.W_q = Layer_Dense(d_model, d_model)
        self.W_k = Layer_Dense(d_model, d_model)
        self.W_v = Layer_Dense(d_model, d_model)
        self.W_o = Layer_Dense(d_model, d_model)

        self.attn_dropout = Dropout(dropout, seed=seed)

    def _split_heads(self, X3: np.ndarray) -> np.ndarray:
        # X3: (B, S, D) -> (B, H, S, d_head)
        B, S, D = X3.shape
        H = self.n_heads
        return X3.reshape(B, S, H, self.d_head).transpose(0, 2, 1, 3)

    def _combine_heads(self, X4: np.ndarray) -> np.ndarray:
        # X4: (B, H, S, d_head) -> (B, S, D)
        B, H, S, Dh = X4.shape
        return X4.transpose(0, 2, 1, 3).reshape(B, S, H * Dh)

    def forward(self, X: np.ndarray, attn_mask: np.ndarray | None = None, training: bool = True) -> np.ndarray:
        """
        X: (S, D) or (B, S, D)
        attn_mask (optional): shape (S, S) or broadcastable to (B, H, S, S)
        """
        X3, squeeze_back = _ensure_3d(X)  # (B,S,D)

        # Linear projections
        Q = self.W_q.forward(X3)  # (B,S,D)
        K = self.W_k.forward(X3)
        V = self.W_v.forward(X3)

        # Split into heads
        Qh = self._split_heads(Q)  # (B,H,S,Dh)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        # Scaled dot-product attention
        # scores: (B,H,S,S)
        scores = np.matmul(Qh, Kh.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)

        if attn_mask is not None:
            # Support (S,S) or (B,1,S,S) or (B,H,S,S)
            scores = scores + attn_mask

        probs = softmax(scores, axis=-1)  # (B,H,S,S)
        probs = self.attn_dropout.forward(probs, training=training)

        out_h = np.matmul(probs, Vh)      # (B,H,S,Dh)
        out   = self._combine_heads(out_h) # (B,S,D)

        # Output projection
        out = self.W_o.forward(out)       # (B,S,D)

        self.output = out[0] if squeeze_back else out
        return self.output

# =========================================================
# Feed-Forward Network (Position-wise)
# =========================================================
class FeedForward:
    """
    Standard Transformer FFN: Dense(d_model -> d_ff) + activation + Dense(d_ff -> d_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, seed: int | None = None, activation: str = "gelu"):
        self.fc1 = Layer_Dense(d_model, d_ff)
        self.fc2 = Layer_Dense(d_ff, d_model)
        self.dropout = Dropout(dropout, seed=seed)
        self.activation = activation.lower()

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, x)
        if self.activation == "gelu":
            # GELU approximation (tanh)
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
    """
    Encoder block (Pre-LayerNorm):
      X -> LN -> MHA -> Dropout -> Residual
        -> LN -> FFN -> Dropout -> Residual
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        seed: int | None = None,
        activation: str = "gelu",
    ):
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        self.mha = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, seed=seed)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout, seed=seed, activation=activation)

        self.drop1 = Dropout(dropout, seed=seed)
        self.drop2 = Dropout(dropout, seed=seed)

    def forward(self, X: np.ndarray, attn_mask: np.ndarray | None = None, training: bool = True) -> np.ndarray:
        X3, squeeze_back = _ensure_3d(X)

        # Attention sublayer (Pre-LN)
        h1 = self.ln1.forward(X3)
        a  = self.mha.forward(h1, attn_mask=attn_mask, training=training)
        a  = self.drop1.forward(a, training=training)
        X3 = X3 + a

        # FFN sublayer (Pre-LN)
        h2 = self.ln2.forward(X3)
        f  = self.ffn.forward(h2, training=training)
        f  = self.drop2.forward(f, training=training)
        X3 = X3 + f

        self.output = X3[0] if squeeze_back else X3
        return self.output

# =========================================================
# Transformer Encoder (stack of layers)
# =========================================================
class TransformerEncoder:
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        seed: int | None = None,
        activation: str = "gelu",
    ):
        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=dropout, seed=seed, activation=activation)
            for _ in range(n_layers)
        ]

    def forward(self, X: np.ndarray, attn_mask: np.ndarray | None = None, training: bool = True) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out, attn_mask=attn_mask, training=training)
        self.output = out
        return out

