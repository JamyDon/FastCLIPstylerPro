import torch
import torch.nn as nn
import torch.optim as optim
from styleaug.text_embedder import TextEmbedder

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),  # 输出判别 [0, 1]
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = TextEmbedder('fastclipstyler').to(device)
discriminator = Discriminator(input_dim=100).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

adversarial_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

num_epochs = 50
batch_size = 64

for epoch in range(num_epochs):
    for batch_idx, (text_embedding, styler_embedding) in enumerate(dataloader):
        text_embedding = text_embedding.to(device)
        styler_embedding = styler_embedding.to(device)
        optimizer_D.zero_grad()

        # 真实数据损失
        real_labels = torch.ones((batch_size, 1), device=device)  # 标记真实样本
        real_loss = adversarial_loss(discriminator(styler_embedding), real_labels)

        # 生成的样本 (假的 styler_embedding)
        fake_styler_embedding = generator(text_embedding).detach()  # 生成假样本
        fake_labels = torch.zeros((batch_size, 1), device=device)  # 标记生成样本
        fake_loss = adversarial_loss(discriminator(fake_styler_embedding), fake_labels)

        # 判别器的总损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()


        optimizer_G.zero_grad()

        # 生成的样本 (使用生成器产生的样本)
        fake_styler_embedding = generator(text_embedding)

        # 生成器的对抗损失（欺骗判别器，使判别器认为生成样本为真）
        g_adversarial_loss = adversarial_loss(discriminator(fake_styler_embedding), real_labels)

        # 回归任务的 MSE 损失：生成的 `styler_embedding` 和真实的 `styler_embedding` 接近
        g_regression_loss = mse_loss(fake_styler_embedding, styler_embedding)

        # 生成器的总损失
        g_loss = g_adversarial_loss + 0.5 * g_regression_loss
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], "
                  f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')