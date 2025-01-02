import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from modules.wikiart import WikiArtDataset
from modules.utils import detect_platform, CLIParser
from modules.autoencoder import AutoEncoder
from torch.optim import lr_scheduler

# Embeddings module
class StyleEmbeddings(nn.Module):
    """
    Cf. here: https://github.com/heejin928/How-Positive-Are-You-Text-Style-Transfer-using-Adaptive-Style-Embedding/blob/master/model.py#L139
    """
    def __init__(self, n_style, d_style):
        super(StyleEmbeddings, self).__init__()
        self.embds = nn.Embedding(n_style, d_style)

    def forward(self, x):
        return self.embds(x)

# Discriminator
class Discriminator(nn.Module):
    """
    Cf. here: https://github.com/heejin928/How-Positive-Are-You-Text-Style-Transfer-using-Adaptive-Style-Embedding/blob/master/model.py#L594
    """
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.embds = StyleEmbeddings(n_classes, 32)
        self.fc1 = nn.Linear(n_classes, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.fc1(input)
        out = self.softmax(out)
        return out  # batch_size * label_size

    def getFc(self, input):
        return self.fc1(input)

    def getSig(self, input):
        return self.softmax(input)
    
    def getSim(self, latent, style=None):
        #latent_norm=torch.norm(latent, dim=-1) #batch, dim
        # Clone, because we don't want the update to happen in latent space
        latent_clone=latent.clone()

        # Get the relevant style embedding for each sample in batch
        style=torch.tensor([[[i] for i in range(self.n_classes)]]).long().to(device)
        style=torch.cat(latent.size(0)*[style]) #128, 2, 1
        style=style.reshape(latent_clone.size(0), -1, 1)
        style=self.embds(style) #(batch. style_num, 1, dim)
        style=style.reshape(style.size(0), style.size(1), -1)
    
        # Binary measure: Does input correspond to style?
        # 1 => yes, 0 => no depending on cosine similarity
        # Use torch.bmm between style embeddings
        # and encoder embeddings.
        dot=torch.bmm(style, latent_clone.unsqueeze(2)) #batch, style_num, 1
        dot=dot.reshape(dot.size(0), dot.size(1))
        return style, dot

if __name__ == "__main__":
    # get arguments
    cp = CLIParser()
    ap = cp.get_argument_parser()
    args = ap.parse_args()
    
    print(args)

    autoencoder_losses = []
    discriminator_losses = []
    discriminator_acc = []

    learning_rate = getattr(args, "training_discriminator.learning_rate")
    n_epochs = getattr(args, "training_discriminator.epochs")
    save_dir = args.model_save_dir
    trainingdir = args.train_folder
    testingdir = args.test_folder
    discriminator_name = args.discriminator_model_name
    device = detect_platform(args.cuda_num)
    batch_size=args.batch_size

    # LOAD TEST DATASET
    traindataset = WikiArtDataset(trainingdir, device)

    # Training the style embeddings
    # We use BCELoss as criterion.
    discriminator = Discriminator(len(traindataset.classes)).to(device)
    dis_criterion=torch.nn.CrossEntropyLoss()
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    autoencoder = AutoEncoder().to(device)
    loss_ = nn.MSELoss()
    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Use learning rate decay - reduce learning rate as training continues
    scheduler = lr_scheduler.StepLR(optimizer_ae, 
                                    step_size=100,
                                    gamma=0.1)
    
    # Initialise dataset, sampler, etc. (balanced)
    weights = traindataset.get_balancing_weights()
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
    train_loader_balanced = torch.utils.data.DataLoader(
        traindataset, 
        batch_size = batch_size,
        sampler = sampler, 
        drop_last = True,
        ) 
    
    # main training loop
    eval_every = 10
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            for batch in tqdm(train_loader_balanced):
                # Inspired by: https://github.com/heejin928/How-Positive-Are-You-Text-Style-Transfer-using-Adaptive-Style-Embedding/blob/master/train.py#L10
                ### LABELS ###
                lbls = batch[1]
                lbls_unsq = lbls.unsqueeze(1)

                # convert labels into one-hot format
                y_onehot = torch.FloatTensor(batch_size, discriminator.n_classes)
                y_onehot.zero_()
                y_onehot = y_onehot.float()
                y_onehot.scatter_(1, lbls_unsq, 1)

                ### GET ENCODED INPUTS ###
                inputs = batch[0].to(device)
                encoded = autoencoder.encoder(inputs)
                # encoded = encoded.cpu().detach() # these must not be updated

                ### GET SIMILARITY ###
                style, similarity=discriminator.getSim(encoded) #style (128, 2, 256), sim(128, 2)
                dis_out=discriminator.forward(similarity)
                max_idx = torch.argmax(dis_out, 1, keepdim=True)
                one_hot = torch.FloatTensor(dis_out.shape).to(device)
                one_hot.zero_()
                one_hot.scatter_(1, max_idx, 1)
                style_pred = one_hot
                style_emb=style.clone()[torch.arange(style.size(0)), lbls.squeeze().long()] #(128, 256)
                style_loss=dis_criterion(dis_out.to(device), y_onehot.to(device))

                ### STYLE-AUGMENTED AUTOENCODER FORWARD PASS ###
                add_latent=encoded+style_emb #batch, dim
                out=autoencoder.decoder(add_latent)
                loss_rec=loss_(out, inputs)

                ### COMBINED LOSS AND BACKPROPAGATION
                pred=style_pred.to('cpu').detach()
                pred = torch.argmax(pred, dim=1).tolist()
                true=lbls.squeeze().to('cpu').tolist()

                dis_acc=accuracy_score(pred, true)
                discriminator_acc.append(dis_acc)
                loss=loss_rec+style_loss

                optimizer_ae.zero_grad()
                optimizer_dis.zero_grad()
                loss.backward()
                optimizer_ae.step()
                optimizer_dis.step()

                autoencoder_losses.append(loss_rec.item())
                discriminator_losses.append(style_loss.item())
 
    # Save the model
    torch.save(discriminator.state_dict(), f"{save_dir}/{ discriminator_name }")