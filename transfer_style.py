import os
from modules.autoencoder import AutoEncoder
from train_discriminator import Discriminator
from modules.utils import detect_platform, CLIParser
from modules.wikiart import WikiArtDataset
import torchvision.transforms as T
from torchvision.io import read_image
import torch

def style_augmented_image(traindataset, autoencoder, style_embeddings, image, original_style, artstyle):
    idx_new = traindataset.classes.index(artstyle)
    idx_original = traindataset.classes.index(original_style)
    inputs = image.unsqueeze(0) 
    encoded_repr = autoencoder.encoder(inputs)
    mod_enc_repr = encoded_repr - style_embeddings[idx_original] + style_embeddings[idx_new]
    return autoencoder.decoder(mod_enc_repr)

if __name__ == "__main__":
    # get arguments
    cp = CLIParser()
    ap = cp.get_argument_parser()
    args_old = ap.parse_known_args()

    # Add additional CLI-arguments specifically for style transfer
    ap.add_argument("target_image", type=str, default="0")
    ap.add_argument("original_style", type=str, default="0")
    ap.add_argument("target_style", type=str, default="0")
    args = ap.parse_args()

    autoencoder_path = f"{ args.model_save_dir }/{ args.autoencoder_model_name }"
    discriminator_path = f"{ args.model_save_dir }/{ args.discriminator_model_name }"
    device = detect_platform(args.cuda_num)
    transform = T.ToPILImage()
    trainingdir = args.train_folder
    imgdir = args.image_dir
    os.makedirs(imgdir, exist_ok=True) 
    original_style = args.original_style
    target_style = args.target_style

    # Load dataset
    traindataset = WikiArtDataset(trainingdir, device)

    # Load both models
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(autoencoder_path, weights_only=True))
    autoencoder.to(device)
    autoencoder.eval()

    discriminator = Discriminator(len(traindataset.classes)) 
    discriminator.load_state_dict(torch.load(discriminator_path, weights_only=True))
    discriminator.to(device)
    discriminator.eval()

    # Load image to be processed
    image = read_image(args.target_image).float()
    print("Image shape: ", image.shape)
    # Ensure the fed image has the right size
    image = T.Resize((416, 416))(image).to(device)

    print("Current image class:", original_style)

    img = transform(style_augmented_image(
        traindataset,
        autoencoder,
        discriminator.embds.embds.weight.data,
        image,
        original_style,
        target_style).squeeze())

    img.save(f"{ imgdir }/transferred_image.png", "PNG")