import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import clip
import os
from styleaug.text_embedder import TextEmbedder
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TensorTextDataset(Dataset):
    def __init__(self, tensor_files, text_files):
        assert len(tensor_files) == len(text_files), "Tensor files and text files must have the same length!"
        self.tensor_files = tensor_files
        self.text_files = text_files

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor = np.load(self.tensor_files[idx])
        with open(self.text_files[idx], 'r') as f:
            text = f.read().strip().split('\n')
        return torch.tensor(tensor, dtype=torch.float32), text


class IndividualTensorTextDataset(Dataset):
    def __init__(self, tensors, texts):
        assert len(tensors) == len(texts), "Number of tensors and texts must match!"
        self.data = []
        seen_texts = set()
        for tensor, task_texts in zip(tensors, texts):
            for i in range(tensor.shape[0]):
                if task_texts[i] not in seen_texts:
                    self.data.append((tensor[i].unsqueeze(0), task_texts[i]))
                    seen_texts.add(task_texts[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    tensors = []
    texts = []
    for tensor, text in batch:
        tensors.append(tensor)
        texts.append(text)
    tensors = torch.cat(tensors, dim=0)
    return tensors, texts

input_dir = Path("med_data/")
tensor_files = sorted(input_dir.glob('tensor_*.npy'))
text_files = sorted(input_dir.glob('text_*.txt'))
assert len(tensor_files) > 0 and len(text_files) > 0, "No tensor or text files found in the directory!"

dataset = TensorTextDataset(tensor_files, text_files)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

all_tensors = []
all_texts = []
for tensors, texts in dataloader:
    all_tensors.append(tensors.squeeze(0))
    all_texts.append(texts[0])

flattened_dataset = IndividualTensorTextDataset(all_tensors, all_texts)

train_size = int(0.9 * len(flattened_dataset))
val_size = len(flattened_dataset) - train_size
train_dataset, val_dataset = random_split(flattened_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

clip_model, _ = clip.load('ViT-B/32', device, jit=False)
clip_model.to(device)
clip_model.eval()
clip_model.requires_grad_(False)

text_styler_embedder = TextEmbedder('fastclipstyler').to(device)
optimizer = optim.Adam(text_styler_embedder.parameters(), lr=1e-4, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

mse_loss = nn.MSELoss()

temperature = 0.1
num_epochs = 2
alpha = 1.0
beta = 0.1

mse_losses = []
contrastive_losses = []
val_mse_losses = []
val_contrastive_losses = []

best_val_loss = float('inf')


def contrastive_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    similarity_matrix = torch.mm(z1, z2.T)

    batch_size = z1.size(0)
    labels = torch.arange(batch_size).to(device)

    logits = similarity_matrix / temperature

    loss = F.cross_entropy(logits, labels)

    return loss


# 训练和验证过程
for epoch in range(num_epochs):
    # 训练阶段
    text_styler_embedder.train()
    epoch_mse_loss = 0
    epoch_contrastive_loss = 0

    for batch_idx, (tensors, texts) in enumerate(train_dataloader):
        text_embeddings = clip_model.encode_text(clip.tokenize(texts).to(device))
        optimizer.zero_grad()

        true_styler_embedding = tensors.squeeze(1)
        fake_styler_embedding = text_styler_embedder(text_embeddings)

        reconstruction_loss = mse_loss(fake_styler_embedding, true_styler_embedding)
        contrastive_loss_value = contrastive_loss(fake_styler_embedding, true_styler_embedding, temperature)
        total_loss = alpha * reconstruction_loss + beta * contrastive_loss_value
        total_loss.backward()
        optimizer.step()

        epoch_mse_loss += reconstruction_loss.item()
        epoch_contrastive_loss += contrastive_loss_value.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], "
                  f"Reconstruction Loss: {reconstruction_loss.item():.4f}, "
                  f"Contrastive Loss: {contrastive_loss_value.item():.4f}, "
                  f"Total Loss: {total_loss.item():.4f}")

    mse_losses.append(epoch_mse_loss / len(train_dataloader))
    contrastive_losses.append(epoch_contrastive_loss / len(train_dataloader))

    text_styler_embedder.eval()
    val_mse_loss = 0
    val_contrastive_loss = 0

    with torch.no_grad(): 
        for tensors, texts in val_dataloader:
            text_embeddings = clip_model.encode_text(clip.tokenize(texts).to(device))

            true_styler_embedding = text_styler_standard(text_embeddings)
            fake_styler_embedding = text_styler_embedder(text_embeddings)

            reconstruction_loss = mse_loss(fake_styler_embedding, true_styler_embedding)
            contrastive_loss_value = contrastive_loss(fake_styler_embedding, true_styler_embedding, temperature)

            val_mse_loss += reconstruction_loss.item()
            val_contrastive_loss += contrastive_loss_value.item()

    val_mse_loss /= len(val_dataloader)
    val_contrastive_loss /= len(val_dataloader)
    val_mse_losses.append(val_mse_loss)
    val_contrastive_losses.append(val_contrastive_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Mean Reconstruction Loss: {mse_losses[-1]:.4f}, "
          f"Train Mean Contrastive Loss: {contrastive_losses[-1]:.4f}, "
          f"Val Mean Reconstruction Loss: {val_mse_loss:.4f}, "
          f"Val Mean Contrastive Loss: {val_contrastive_loss:.4f}")

    val_total_loss = alpha * val_mse_loss + beta * val_contrastive_loss
    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        torch.save(text_styler_embedder.state_dict(), 'best_text_styler_embedder.pth')
        print("Saved Best Model with Val Loss:", best_val_loss)

    scheduler.step()

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(mse_losses, label='Train MSE Loss', color='blue')
plt.plot(contrastive_losses, label='Train Contrastive Loss', color='red')
plt.plot(val_mse_losses, label='Val MSE Loss', color='orange')
plt.plot(val_contrastive_losses, label='Val Contrastive Loss', color='green')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Trends")
plt.legend()
plt.grid()

plt.savefig(f"{output_dir}/loss_trend_plot.png", dpi=300)
plt.show()