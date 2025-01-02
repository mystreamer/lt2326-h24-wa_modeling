import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from modules.wikiart import WikiArtDataset
from modules.utils import detect_platform
from modules.autoencoder import AutoEncoder
from torch.optim import lr_scheduler

# Embeddings module
class StyleEmbeddings(nn.Module):
    def __init__(self, n_style, d_style):
        super(StyleEmbeddings, self).__init__()
        self.embds = nn.Embedding(n_style, d_style)

    def forward(self, x):
        return self.embds(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # self.gpu=gpu
        self.n_classes = n_classes
        self.embds = StyleEmbeddings(n_classes, 32)
        self.fc1 = nn.Linear(n_classes, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.fc1(input)
        out = self.softmax(out)

        # out = F.log_softmax(out, dim=1)
        return out  # batch_size * label_size

    def getFc(self, input):
        return self.fc1(input)

    def getSig(self, input):
        return self.softmax(input)
    
    def getSim(self, latent, style=None):
        #latent_norm=torch.norm(latent, dim=-1) #batch, dim
        # Clone, because we don't want the update to happen in latent space
        latent_clone=latent.clone()
        # if style is not None:
        #     style=style.unsqueeze(2)
        #     style=self.style_embed(style.long())
        #     # pdb.set_trace()
        #     style=style.reshape(style.size(0), style.size(1), style.size(-1))
        # else:
        style=torch.tensor([[[i] for i in range(self.n_classes)]]).long().to(device)
        style=torch.cat(latent.size(0)*[style]) #128, 2, 1
        style=style.reshape(latent_clone.size(0), -1, 1)
        # get the according style embedding for each sample in batch
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
    autoencoder_losses = []
    discriminator_losses = []
    discriminator_acc = []

    LEARNING_RATE = 0.1
    CUDA_NUM = 0
    SAVE_DIR = "./models"

    # Dataset paths
    trainingdir = "/Users/dylan/Downloads/wikiart/train"
    testingdir = "/Users/dylan/Downloads/wikiart/test"

    device = detect_platform(CUDA_NUM)
    batch_size=32

    # LOAD TEST DATASET
    traindataset = WikiArtDataset(trainingdir, device)

    # Training the style embeddings
    # We use BCELoss as criterion.
    discriminator = Discriminator(len(traindataset.classes)).to(device)
    dis_criterion=torch.nn.CrossEntropyLoss()
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    autoencoder = AutoEncoder().to(device)
    loss_ = nn.MSELoss()
    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

    # Use learning rate decay - reduce learning rate as training continues
    scheduler = lr_scheduler.StepLR(optimizer_ae, 
                                    step_size=100,
                                    gamma=0.1)
    

    weights = traindataset.get_balancing_weights()
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
    train_loader_balanced = torch.utils.data.DataLoader(
        traindataset, 
        batch_size = 32,
        sampler = sampler, 
        drop_last = True,
        ) 

    n_epochs = 5
    eval_every = 10
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            for batch in tqdm(train_loader_balanced):
                ### LABELS ###
                lbls = batch[1]
                lbls_unsq = lbls.unsqueeze(1)
                y_onehot = torch.FloatTensor(32, discriminator.n_classes)
                # In your for loop
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
                # one=torch.tensor(1)
                # zero=torch.tensor(0)
                # style_pred=torch.where(dis_out>0.5, one, zero)
                max_idx = torch.argmax(dis_out, 1, keepdim=True)
                # max_idx = max_idx.squeeze(0).unsqueeze(1)
                one_hot = torch.FloatTensor(dis_out.shape).to(device)
                one_hot.zero_()
                one_hot.scatter_(1, max_idx, 1)
                # style_pred=style_pred.reshape(one_hot.size(0))
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

                # import pdb; pdb.set_trace()
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
    torch.save(discriminator.state_dict(), f"{SAVE_DIR}/discriminator.pth")