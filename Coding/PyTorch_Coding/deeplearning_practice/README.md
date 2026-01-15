## PyTorch Quick Notes

### Convert Python data to a PyTorch tensor
Use `torch.tensor()` to convert Python data (int, float, list, etc.) into a PyTorch tensor:

```python
import torch
x = torch.tensor(x)
```
### Converting to specific tensor data type:
```python
x = torch.tensor(x, dtype=torch.float)
```
- Other common datatypes:
- torch.int
- torch.long
- torch.float32
- torch.float64

### Sigmoid activation
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

### Softmax Activation Function
\(\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}\)

### ReLU Activation
\[
\text{ReLU}(x) = \max(0, x)
\]
Piecewise form:
\[
\text{ReLU}(x) =
\begin{cases}
0, & x \le 0 \\
x, & x > 0
\end{cases}
\]

### Tanh Activation
\[
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
\]
Equivalent form:
\[
\tanh(x) = \frac{2}{1 + e^{-2x}} - 1
\]






