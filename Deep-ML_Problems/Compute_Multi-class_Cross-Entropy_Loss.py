# Problem 134 Compute Multi-class Cross-Entropy Loss
import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    cross_entropy = np.sum(np.multiply(predicted_probs, true_labels), axis=-1)
    loss = []
    for i in cross_entropy:
        loss.append(i + epsilon)

    cross_loss = -np.mean(np.log(loss))

    return max(cross_loss, 0.0)