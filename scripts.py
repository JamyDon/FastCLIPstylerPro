import os
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from styleaug.text_embedder import TextEmbedder
from FastCLIPstyler import FastCLIPStyler


class params:
    img_width = 512
    img_height = 512
    num_crops = 16
    text_encoder = 'fastclipstyler'

class TrainStylePredictor():
    def __init__(self, trainer):
        self.trainer = trainer

    def test(self, content_img, style_desc):
        self.trainer.content_image = content_img
        params.text = style_desc
        return self.trainer.test()

def style_transfer(trainer, content_img, style_description):
    predictor = TrainStylePredictor(trainer)
    styled_image = predictor.test(content_img, style_description)
    return styled_image

def process_images_in_folder(trainer, input_folder, output_folder, style_description):
    os.makedirs(output_folder, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((params.img_height, params.img_width))
    ])
    to_pil = transforms.ToPILImage()

    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    if len(all_files) == 0:
        print(f"No valid images found in input folder: {input_folder}")
        return

    for img_file in tqdm(all_files, desc="Processing images"):
        input_image_path = os.path.join(input_folder, img_file)
        output_image_path = os.path.join(output_folder, img_file)

        try:
            image = Image.open(input_image_path).convert("RGB")
            input_image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                styled_image = style_transfer(trainer, input_image, style_description)[0]
                styled_image = to_pil(styled_image)
            styled_image.save(output_image_path)
        except Exception as e:
            print(f"Error processing file {img_file}: {e}")

    print(f"Processing completed. Styled images are saved in '{output_folder}'.")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

if __name__ == "__main__":
    print("Initializing model...")
    fastclipstyler_trainer = FastCLIPStyler(params)
    input_folder = "new_input"
    output_folder = "output_images_old"
    style_description = "deep dark fantasy"
    output_folder += style_description
    process_images_in_folder(fastclipstyler_trainer, input_folder, output_folder, style_description)