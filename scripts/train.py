import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ASVSpoofDataset
from model import RawNet2
import argparse
from tqdm import tqdm
import os


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train Epoch: {epoch}\tLoss: {avg_loss:.4f}\tAccuracy: {accuracy:.2f}%')


def validate(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation: Loss: {avg_loss:.4f}\tAccuracy: {accuracy:.2f}%')
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_root', type=str, default='data')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize paths for Windows
    train_dir = os.path.normpath(os.path.join(args.data_root, 'train'))
    dev_dir = os.path.normpath(os.path.join(args.data_root, 'dev'))

    print(f"\nLoading datasets from:\n- Train: {train_dir}\n- Dev: {dev_dir}")

    # Create datasets with verification
    try:
        train_dataset = ASVSpoofDataset(train_dir)
        val_dataset = ASVSpoofDataset(dev_dir)
    except Exception as e:
        print(f"\nError loading datasets: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify the dataset files exist in the correct locations")
        print("2. Check the directory structure matches:")
        print("   data/train/bonafide/")
        print("   data/train/spoof/")
        print("3. Ensure files have .flac extension")
        print("4. Run organize_data.py if files aren't properly sorted")
        return

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    # Initialize model
    model = RawNet2().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        val_acc = validate(model, device, val_loader, criterion)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with accuracy {best_acc:.2f}%")

    print(f'\nTraining complete. Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()