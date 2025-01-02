import os
from modules.autoencoder import AutoEncoder
from train_discriminator import Discriminator
from modules.utils import detect_platform
from modules.wikiart import WikiArtDataset
import torch
import torchvision.transforms as T

def style_augmented_image(traindataset, autoencoder, style_embeddings, image, original_style, artstyle):
    idx_new = traindataset.classes.index(artstyle)
    idx_original = traindataset.classes.index(original_style)
    # import pdb; pdb.set_trace()
    # inputs, classes = next(iter(train_loader_balanced))  
    inputs = image.unsqueeze(0) 
    encoded_repr = autoencoder.encoder(inputs)
    mod_enc_repr = encoded_repr - style_embeddings[idx_original] + style_embeddings[idx_new]
    return autoencoder.decoder(mod_enc_repr)

if __name__ == "__main__":
    CUDA_NUM = 0
    print("Another test...")
    AUTOENCODER_PATH = "./models/autoencoder.pth"
    DISCRIMINATOR_PATH = "./models/discriminator.pth"

    device = detect_platform(CUDA_NUM)
    transform = T.ToPILImage()

    trainingdir = "/Users/dylan/Downloads/wikiart/train"

    # Load dataset
    traindataset = WikiArtDataset(trainingdir, device)

    # Load both models
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, weights_only=True))
    autoencoder.to(device)
    autoencoder.eval()

    discriminator = Discriminator(len(traindataset.classes)) 
    discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH, weights_only=True))
    discriminator.to(device)
    discriminator.eval()

    IMAGE_NO = 1

    image = traindataset[IMAGE_NO][0]

    print("Current image class:", traindataset.classes[traindataset[IMAGE_NO][1]])

    img = transform(style_augmented_image(
        traindataset,
        autoencoder,
        discriminator.embds.embds.weight.data,
        image,
        traindataset.classes[traindataset[IMAGE_NO][1]],
        "Abstract_Expressionism").squeeze())

    os.makedirs("./images", exist_ok=True) 
    img.save(f"./images/transferred_image.png", "PNG")