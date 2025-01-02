import torch
from torch import nn, Tensor
from typing import Tuple
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np

# local imports
from modules.autoencoder import AutoEncoder
from modules.utils import detect_platform
from modules.wikiart import WikiArtDataset

print(__name__)

if __name__ == "__main__":
    # Learning Rate
    LEARNING_RATE = 0.1
    CUDA_NUM = 0
    SAVE_DIR = "./models"

    # load the dataset
    device = detect_platform(CUDA_NUM)
    batch_size=32

    ##### LOAD DATASTET #####
    # SET PATH
    trainingdir = "/Users/dylan/Downloads/wikiart/train"
    testingdir = "/Users/dylan/Downloads/wikiart/test"
    # LOAD TRAIN DATASET
    traindataset = WikiArtDataset(trainingdir, device)
    # LOAD TEST DATASET
    testdataset = WikiArtDataset(testingdir, device)

    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Activate training mode
    model.train()

    # Use MSE as loss function
    # We use the Mean Squared Error as the reconstruction loss.
    loss_ = nn.MSELoss()

    # Use learning rate decay - reduce learning rate as training continues
    scheduler = lr_scheduler.StepLR(optimizer, 
                                    step_size=100,
                                    gamma=0.1)

    # from tqdm.notebook import tqdm
    # import numpy as np
    weights = traindataset.get_balancing_weights()
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     

    train_loader_balanced = torch.utils.data.DataLoader(
        traindataset, 
        batch_size = batch_size,
        sampler = sampler, 
    ) 

    test_loader = torch.utils.data.DataLoader(
        testdataset,
        batch_size=batch_size,
        shuffle=True,
    )

    ### TRAINING ###
    n_epochs = 5
    eval_every = 10
    best_loss = np.infty

    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            losses = []
            # Iterate over data in batches
            for x_batch, y_batch in tqdm(train_loader_balanced, leave=False):
                # PyTorch specific; We need to reset all gradients
                optimizer.zero_grad()
                
                # 0. Transform input batch data from 28 X 28 to 784 features
                #   Note that our encoder maps the data into just 10 features!
                x_batch = x_batch.to(device)
                # x_batch = x_batch.view(x_batch.shape[0], -1)
                # x_batch = x_batch[0]
                # print(x_batch.shape)
                
                # 1. Apply AutoEncoder model (forward pass).
                #    We use the output of the decoder for training.
                output = model(x_batch)[1]
                
                # 2. Calculate the reconstruction loss
                loss = loss_(output, x_batch)
                losses.append(loss.item())
                
                # 3. Backpropagate the loss
                loss.backward()
                
                # 4. Update the weights
                optimizer.step()
            
            # Mean loss of the batches in this epoch
            mean_loss = np.round(np.mean(losses), 5)
            
            # Print current loss after 'eval_every' epochs
            if (epoch+1) % eval_every == 0:
                print(f"Loss at epoch [{epoch+1} / {n_epochs}]: {mean_loss}")

            # Update learning rate as training continues
            scheduler.step()

            # Progress bar
            pbar.write('processed: %d' %epoch)
            pbar.update(1) 

    # Save the model
    torch.save(model.state_dict(), f"{SAVE_DIR}/autoencoder.pth")