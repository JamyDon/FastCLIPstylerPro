import torch
import torch.nn as nn
import torch.optim as optim
from styleaug.text_embedder import TextEmbedder
import torch.nn.functional as F
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_styler_embedder = TextEmbedder('fastclipstyler').to(device)
optimizer = optim.Adam(text_styler_embedder.parameters(), lr=1e-4, betas=(0.5, 0.999))
mse_loss = nn.MSELoss()
temperature = 0.07
num_epochs = 50
batch_size = 64
clip_model, _ = clip.load('ViT-B/32', device, jit=False)
clip_model.to(device)
clip_model.requires_grad_(False)
tokens = clip.tokenize([text_prompt[0]]).to(device)


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
    for batch_idx, (text, styler_embedding) in enumerate(dataloader):
        
        text_embedding = text_embedding.to(device)
        styler_embedding = styler_embedding.to(device)
        optimizer.zero_grad()
        fake_styler_embedding = text_styler_embedder(text_embedding)
        reconstruction_loss = mse_loss(fake_styler_embedding, styler_embedding)
        contrastive_loss_value = contrastive_loss(fake_styler_embedding, styler_embedding, temperature)
        total_loss = reconstruction_loss + contrastive_loss_value
        total_loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], "
                  f"Reconstruction Loss: {reconstruction_loss.item():.4f}, "
                  f"Contrastive Loss: {contrastive_loss_value.item():.4f}, "
                  f"Total Loss: {total_loss.item():.4f}")

torch.save(text_styler_embedder.state_dict(), 'text_styler_embedder_with_contrastive_loss.pth')