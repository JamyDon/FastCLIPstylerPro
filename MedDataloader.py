import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

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
final_dataloader = DataLoader(flattened_dataset, batch_size=10, shuffle=True)

for batch_idx, (tensors_batch, texts_batch) in enumerate(final_dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  Tensor batch shape: {tensors_batch.shape}")
    print(f"  Text batch: {texts_batch}")
    input()