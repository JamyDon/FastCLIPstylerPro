import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import clip
from styleaug.text_embedder import TextEmbedder

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
        for tensor, task_texts in zip(tensors, texts):
            for i in range(tensor.shape[0]):
                self.data.append((tensor[i].unsqueeze(0), task_texts[i]))

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
final_dataloader = DataLoader(flattened_dataset, batch_size=16, shuffle=True)

clip_model, _ = clip.load('ViT-B/32', device, jit=False)
clip_model.to(device)
clip_model.requires_grad_(False)

text_styler_embedder = TextEmbedder('fastclipstyler').to(device)

optimizer = optim.Adam(text_styler_embedder.parameters(), lr=1e-4, betas=(0.5, 0.999))
mse_loss = nn.MSELoss()

temperature = 0.07
num_epochs = 50

def contrastive_loss(z1, z2, temperature=0.07):
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    similarity_matrix = torch.mm(z1, z2.T)
    positive_indices = torch.arange(batch_size).to(device)
    labels = positive_indices
    logits = similarity_matrix / temperature
    loss = F.cross_entropy(logits, labels)
    return loss

for epoch in range(num_epochs):
    text_styler_embedder.train()
    for batch_idx, (tensors, texts) in enumerate(final_dataloader):
        tensors = tensors.to(device)
        text_embeddings = clip_model.encode_text(clip.tokenize(texts).to(device))
        text_embeddings = F.normalize(text_embeddings, dim=1)
        optimizer.zero_grad()
        fake_styler_embedding = text_styler_embedder(text_embeddings)
        
        # MSE
        reconstruction_loss = mse_loss(fake_styler_embedding, tensors.squeeze(1))
        
        # Contrastive
        contrastive_loss_value = contrastive_loss(fake_styler_embedding, tensors.squeeze(1), temperature)
        
        total_loss = reconstruction_loss + contrastive_loss_value
        total_loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], "
                  f"Reconstruction Loss: {reconstruction_loss.item():.4f}, "
                  f"Contrastive Loss: {contrastive_loss_value.item():.4f}, "
                  f"Total Loss: {total_loss.item():.4f}")

torch.save(text_styler_embedder.state_dict(), 'text_styler_embedder_with_contrastive_loss.pth')