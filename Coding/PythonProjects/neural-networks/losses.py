import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
#CategoricalCrossentropy and Multi-class Cross Entrpy
class Loss_CategoricalCrossentropy(Loss): #-ve sum of multiplication of true and log function of predicted
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #if y < 1e-7, then y = 1e-7. If y > 1- 1e-7, then y=1-1e-7
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Binary Cross-Entropy (BCE)
class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Flatten to (N,)
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        # Clip for numerical stability
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Per-sample BCE
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
# MSE Loss
class Loss_MSE(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2, axis=1)

#MAE loss
class Loss_MAE(Loss):
    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred), axis=1)


