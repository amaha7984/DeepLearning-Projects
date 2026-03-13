# This folder contains the information about beural networks

### Model Optimization
- Distillation: Train a smaller model (student) to mimic a larger model (teacher)
- Pruning: Remove unimportant weights or neurons.
- Quantization: Reduce numerical precision (FP32 → FP16 → INT8 → INT4) or conversion of higher memory format to a lower memory format
    - FP32 (float32) -> Single / Full Precision
    - FP16 and BF16 ->  Half Precision
- LoRA (Low-Rank Adaptation):
 Instead of updating all the large model weights during fine-tuning, LoRA freezes the original model and adds small trainable low-rank matrices on top of certain layers
- QLoRA = Quantized LoRA
It combines:
    - Quantization (store the base model in low precision, like 4-bit)
    - LoRA adapters (small trainable matrices)
- Continual learning:
Training a model sequentially on new tasks or new data without forgetting what it previously learned.
Example:
    - Train model on Task A
    - Later train on Task B
    - Model should perform well on both A and B

### Optimization Algorithms
#### Gradient Descent:
-	Gradient descent is an iterative optimization algorithm that minimizes a loss function by updating parameters (weights) in the negative direction of the gradient, 
where the gradient is the derivative of the loss with respect to the parameters. The update is controlled by the learning rate.
-	Update rule: w = w – η ∇L, 
Where: w = weights, η = learning rate, ∇L = gradient

#### Batch Gradient Descent → uses full dataset

#### Stochastic Gradient Descent (SGD) → one sample at a time

#### Mini-batch Gradient Descent → small batch (most common in deep learning)










    

















