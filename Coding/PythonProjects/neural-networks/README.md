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






    

















