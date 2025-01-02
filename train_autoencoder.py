import torch
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np

# local imports
from modules.autoencoder import AutoEncoder
from modules.utils import detect_platform, CLIParser
from modules.wikiart import WikiArtDataset

print(__name__)

if __name__ == "__main__":
    # get arguments
    cp = CLIParser()
    ap = cp.get_argument_parser()
    args = ap.parse_args()

    # print(args)

    # Set params from config / CLI
    learning_rate = getattr(args, "training_autoencoder.learning_rate")
    n_epochs = getattr(args, "training_autoencoder.epochs")
    traindir = args.train_folder
    testdir = args.test_folder
    autoencoder_name = args.autoencoder_model_name
    device = detect_platform(args.cuda_num)
    save_dir = args.model_save_dir
    batch_size = args.batch_size

    # Load datasets
    traindataset = WikiArtDataset(traindir, device)
    testdataset = WikiArtDataset(testdir, device)

    # Init model & optimiser
    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Ensure model is ready to train (gradients active)
    model.train()

    # Use MSE as loss function
    # We use the Mean Squared Error as the reconstruction loss.
    loss_ = nn.MSELoss()

    # Use learning rate decay - reduce learning rate as training continues
    scheduler = lr_scheduler.StepLR(optimizer, 
                                    step_size=100,
                                    gamma=0.1)
    
    # Perform dataset balancing
    weights = traindataset.get_balancing_weights()
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     

    # Create the dataloaders
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
    eval_every = 10
    best_loss = np.infty

    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            losses = []
            # Iterate over data in batches
            for x_batch, y_batch in tqdm(train_loader_balanced, leave=False):
                optimizer.zero_grad()
                
                x_batch = x_batch.to(device)
                
                # forward pass
                output = model(x_batch)[1]
                
                # calc error (reconstruction loss)
                loss = loss_(output, x_batch)
                losses.append(loss.item())
                
                # backprop
                loss.backward()
                
                # update rule
                optimizer.step()
            
            mean_loss = np.round(np.mean(losses), 5)
            
            # evaluate loss
            if (epoch+1) % eval_every == 0:
                print(f"Loss at epoch [{epoch+1} / {n_epochs}]: {mean_loss}")

            # push the scheduler
            scheduler.step()

            # update progress bar
            pbar.write('processed: %d' %epoch)
            pbar.update(1) 

    # save model
    torch.save(model.state_dict(), f"{save_dir}/{ autoencoder_name }")