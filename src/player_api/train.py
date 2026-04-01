import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from player_api.label_cnn import LabelCNN

# python -m player_api.train

DATA_DIR = "player_api/data"
MODEL_DIR = "player_api/models"
PATCH_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)

train_transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
])

class PatchDataset(Dataset):
    def __init__(self, label: str, transform):
        self.transform = transform
        self.samples = []

        pos_dir = os.path.join(DATA_DIR, label, "pos")
        neg_dir = os.path.join(DATA_DIR, label, "neg")

        for f in os.listdir(pos_dir):
            if f.endswith(".png"):
                self.samples.append((os.path.join(pos_dir, f), 1))
        for f in os.listdir(neg_dir):
            if f.endswith(".png"):
                self.samples.append((os.path.join(neg_dir, f), 0))

        n_pos = sum(1 for _, c in self.samples if c == 1)
        n_neg = sum(1 for _, c in self.samples if c == 0)
        print(f"  [{label}] {n_pos} pos, {n_neg} neg")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor([cls], dtype=torch.float32)

def train_model(label: str):
    print(f"\nTraining model for: {label}")

    full_dataset = PatchDataset(label, train_transform)
    n_val = max(1, int(len(full_dataset) * VAL_SPLIT))
    n_train = len(full_dataset) - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
    val_ds.dataset = PatchDataset(label, val_transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = LabelCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    criterion = nn.BCELoss()

    best_acc   = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        train_loss = 0.0
        for patches, labels in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(patches), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for patches, labels in val_dl:
                preds = (model(patches) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"loss: {train_loss/len(train_dl):.4f} | "
              f"val acc: {val_acc:.3f}"
              + (" ← best" if val_acc == best_acc else ""))

    save_path = os.path.join(MODEL_DIR, f"{label}_cnn.pth")
    torch.save(best_state, save_path)
    print(f"  Best val acc: {best_acc:.3f} | Model saved to {save_path}")

if __name__ == "__main__":
    train_model("p1")
    train_model("cpu")