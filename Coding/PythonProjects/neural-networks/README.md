# This folder contains the information about neural networks

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

#### Momentum: 
- Momentum accumulates past gradients to create a velocity term, which accelerates learning in consistent directions and reduces oscillations compared to standard gradient descent.
- Update Rule:
   - v=βv+∇L
   - w=w−ηv, 
   - where  v → called the velocity term (or momentum buffer (accumulated gradients))
   - β (beta) → called the momentum coefficient (or momentum parameter): controls how much past gradient information is retained

#### RMSProp: 
- RMSProp adapts the learning rate per parameter by normalizing the gradient using a moving average of squared gradients, which helps stabilize and speed up convergence.
- Update Rule:
	- Compute moving average of squared gradients:
        - s=βs+(1-β)(∇L)^2
	- Update Weights
        - w = w - (η / (math.sqrt(s) + ϵ)) * ∇L
        
    - Terms
        - s→ running average of squared gradients
        - β→ decay rate (usually ~0.9)
        - η→ learning rate
        - ϵ→ small constant for numerical stability
#### Adam: Adam combines Momentum + RMSProp.
- Uses moving average of gradients → like Momentum
- Uses moving average of squared gradients → like RMSProp
- Adapts learning rate per parameter
- Update Rules:
  - First moment (mean of gradients): m=β_1 m+(1-β_1)∇L
  - Second moment (mean of squared gradients): v=β_2 v+(1-β_2)(∇L)^2
  - 	Bias correction:
m ̂=m/(1-β_1^t ),v ̂=v/(1-β_2^t )
  - Weight update:
  w=w - (η/(√(v ̂ )+ϵ)) * m ̂
- 5.	Terms:
    - η (learning rate) → step size
    - β₁ → momentum coefficient (first moment, usually 0.9)
    - β₂ → decay rate for squared gradients (usually 0.999)
    - ε → small constant for numerical stability (e.g., 1e-8)
    - t → timestep (iteration number, for bias correction)
















    

















