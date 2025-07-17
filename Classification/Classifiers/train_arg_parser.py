import argparse

def get_train_args():
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation dataset')
    parser.add_argument('--model', type=str, default='simple_cnn', help='Model name to use')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='./weights')
    return parser.parse_args()