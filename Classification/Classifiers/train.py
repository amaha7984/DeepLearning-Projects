import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from data import CardDataset, get_transforms
from model import get_model
from train_engine import train_one_epoch, validate
from train_arg_parser import get_train_args

def main():
    args = get_train_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transforms()
    train_loader = DataLoader(CardDataset(args.train_dir, transform), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CardDataset(args.val_dir, transform), batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    best_val_acc = 0.0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        

        torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_epoch_weight.pth'))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_val_acc_weight.pth'))
            print(f"Model saved with best val acc: {best_val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_val_loss_weight.pth'))
            print(f"Model saved  with best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
