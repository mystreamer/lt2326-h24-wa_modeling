import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiArtModel
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]

print("Running...")


traindataset = WikiArtDataset(trainingdir, device)
#testingdataset = WikiArtDataset(testingdir, device)

print(traindataset.imgdir)

the_image, the_label = traindataset[5]
print(the_image, the_image.size())

# the_showable_image = F.to_pil_image(the_image)
# print("Label of img 5 is {}".format(the_label))
# the_showable_image.show()


def train(epochs=3, modelfile=None, device="cpu"):
    loader = DataLoader(traindataset, batch_size=32, shuffle=True)

    model = WikiArtModel().to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss().to(device)
    
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, y = batch
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

model = train(modelfile=config["modelfile"], device=device)
