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
import torcheval.metrics as metrics

testingdir = "../test"
device = "cuda:3"

print("Running...")

#traindataset = WikiArtDataset(trainingdir, device)
testingdataset = WikiArtDataset(testingdir, device)

def test(modelfile=None, device="cpu"):
    loader = DataLoader(traindataset, batch=1)

    model = WikiArtModel()
    model.load_state_dict(torch.load(modelfile, weights_only=True))
    model = model.to(device)
    model.eval()

    predictions = []
    for batch_id, batch in enumerate(tqdm.tqdm(loader)):
        X, y = batch
        y = y.to(device)
        output = model(X)
        predictions.append(output)
        

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model
