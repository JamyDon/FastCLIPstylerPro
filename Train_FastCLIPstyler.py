import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from styleaug.ghiasi import Ghiasi
from styleaug.pbn_embedding import PBNEmbedding
from styleaug.text_embedder import TextEmbedder
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import ToPILImage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
class FastCLIPStyler:

    def __init__(self, opt):

        self.opt = opt

        assert (opt.img_width % 8) == 0, "width must be a multiple of 8"
        assert (opt.img_height % 8) == 0, "height must be a multiple of 8"

        print('Loading Ghiasi model')
        self.styleaug = Ghiasi()
        self.styleaug.to(device)
        self.styleaug.requires_grad_(False)

        print('Loading text embedder')
        self.text_embedder = TextEmbedder(self.opt.text_encoder)
        self.text_embedder.to(device)
        self.text_embedder.requires_grad_(True)

        print('Loading PBN statistics')
        self.pbn_embedder = PBNEmbedding()
        self.pbn_embedder.to(device)
        self.pbn_embedder.requires_grad_(False)

        if opt.text_encoder == 'fastclipstyler':
            print('Loading CLIP')
            self.clip_model, _ = clip.load('ViT-B/32', device, jit=False)
            self.clip_model.to(device)
            self.clip_model.requires_grad_(False)

        elif opt.text_encoder == 'edgeclipstyler':
            print('Loading albert')
            self.bert_encoder = SentenceTransformer('paraphrase-albert-small-v2')

        else:
            raise Exception('Invalid text encoder. Should be either fastclipstyler or edgeclipstyler')

        print('Finished loading all the models')
        print()

        text_source = np.loadtxt('/mlx_devbox/users/wangche.0526/playground/CV/FastCLIPstylerPro/styleaug/checkpoints/source_array.txt')
        self.text_source = torch.Tensor(text_source).to(device)

    def _set_up_features(self):

        with torch.no_grad():

            if self.opt.text_encoder == 'fastclipstyler':

                tokens = clip.tokenize([self.opt.text]).to(device)
                clip_embeddings = self.clip_model.encode_text(tokens).detach()
                clip_embeddings = clip_embeddings.mean(axis=0, keepdim=True)
                clip_embeddings = clip_embeddings.type(torch.float32)
                clip_embeddings /= clip_embeddings.norm(dim=-1, keepdim=True)

                self.text_features = clip_embeddings

            elif self.opt.text_encoder == 'edgeclipstyler':

                bert_embeddings = self.bert_encoder.encode([self.opt.text])
                bert_embeddings = bert_embeddings.mean(axis=0, keepdims=True)
                bert_embeddings /= np.linalg.norm(bert_embeddings)
                bert_embeddings = torch.Tensor(bert_embeddings).to(device)

                self.text_features = bert_embeddings

    def test(self):

        self.text_embedder.load_model()
        self.text_embedder.eval()

        self._set_up_features()
        painting_embedding = self.text_embedder(self.text_features)
        target = self.styleaug(self.content_image, painting_embedding)

        return target


def train_fastclipstyler(opt):
    print("Initializing FastCLIPStyler Training...")
    model = FastCLIPStyler(opt)
    model.text_embedder.train()
    preprocess_clip = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), 
                (0.26862954, 0.26130258, 0.27577711)) 
    ])
    epochs = opt.epochs
    lr = opt.learning_rate
    lambda_dir = opt.lambda_dir
    lambda_patch = opt.lambda_patch
    lambda_dis = opt.lambda_dis

    def directional_loss(image_embedding, text_embedding):
        return -torch.cosine_similarity(image_embedding, text_embedding).mean()

    def patch_clip_loss(image, text_embedding, clip_model, crop_fn):
        patches = crop_fn(image)
        patches = [ToPILImage()(patch) for patch in patches]
        patches = [preprocess_clip(patch) for patch in patches]
        patches = [patch.unsqueeze(0).to(device) for patch in patches]
        patch_embeddings = torch.stack([clip_model.encode_image(patch) for patch in patches])
        return -torch.cosine_similarity(
            patch_embeddings, text_embedding.unsqueeze(0).expand_as(patch_embeddings)
        ).mean()

    def distribution_loss(embedding):
        mu_data = model.pbn_embedder.means  #  [1, 100]
        sigma_data_inv = model.pbn_embedder.cov_inv  #  [100, 100]
        diff = embedding - mu_data  #  [1, 100]
        
        diff_t = diff.T  #  [100, 1]

        maha_distance = torch.matmul(torch.matmul(diff, sigma_data_inv), diff_t) 
        return maha_distance.squeeze()

    optimizer = optim.Adam(model.text_embedder.parameters(), lr=lr)

    train_transform = transforms.Compose([
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.ToTensor()
    ])

    content_images = ["/mlx_devbox/users/wangche.0526/playground/CV/1.png", "/mlx_devbox/users/wangche.0526/playground/CV/2.png"]
    text_prompts = ["Style of cubism", "Style of impressionism"]
    dataset = list(zip(content_images, text_prompts))

    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    for epoch in range(epochs):
        for content_image_path, text_prompt in train_loader:
            # Load image and preprocess
            content_image = Image.open(content_image_path[0]).convert("RGB")
            content_image = train_transform(content_image).unsqueeze(0).to(device)
            model.content_image = content_image

            # Prepare text embedding
            tokens = clip.tokenize([text_prompt[0]]).to(device)
            text_embedding = model.clip_model.encode_text(tokens).detach()

            # Forward pass: Style embedding and stylized image generation
            painting_embedding = model.text_embedder(text_embedding)
            styled_image = model.styleaug(content_image, painting_embedding)

            if styled_image.shape[-2:] != (224, 224):
                styled_image_pil = ToPILImage()(styled_image.squeeze(0)) 
                styled_image_resized = preprocess_clip(styled_image_pil) 
                styled_image_resized = styled_image_resized.unsqueeze(0).to(device) 
            image_embeddings = model.clip_model.encode_image(styled_image_resized)

            # Loss Calculation
            l_dir = directional_loss(image_embeddings, text_embedding)
            l_patch = patch_clip_loss(styled_image_resized, text_embedding, model.clip_model, crop_and_augment)
            l_dis = distribution_loss(painting_embedding)
            print(l_dir)
            print(l_patch)
            print(l_dis)
            total_loss = lambda_dir * l_dir + lambda_patch * l_patch + lambda_dis * l_dis

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item()}")

    torch.save(model.text_embedder.state_dict(), "trained_text_embedder.pth")
    print("Training complete. Model saved as 'trained_text_embedder.pth'.")

def crop_and_augment(image, num_patches=16, crop_size=64):
    patches = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor()
    ])
    for _ in range(num_patches):
        patches.append(transform(image.squeeze(0)))
    print(len(patches))
    return patches


class Opt:
    img_width = 512
    img_height = 512
    text = "Style of cubism"
    text_encoder = "fastclipstyler"
    epochs = 5
    batch_size = 1
    learning_rate = 1e-4
    lambda_dir = 1.0
    lambda_patch = 1.0
    lambda_dis = 0.02

opt = Opt()
train_fastclipstyler(opt)