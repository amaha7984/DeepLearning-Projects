# Different Classifiers Ranging from CNNs to ViTs

- Run Training with:
```bash
python3.9 train.py \
  --train_dir /path/to/train \
  --val_dir /path/to/valid \
  --num_classes 53 \
  --model vit_scratch \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --save_dir ./weights
```
