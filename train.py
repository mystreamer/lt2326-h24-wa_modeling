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

trainingdir = "./train"
testingdir = "./test"

print("Running...")

class WikiArtDataset(Dataset):
    def __init__(self, imgdir):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            for art in artfiles:
                filedict[art] = arttype
                indices.append(art)
        print("...finished")
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        
    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        label = self.filedict[imgname]
        image = read_image(os.path.join(self.imgdir, label, imgname))

        return image, label

traindataset = WikiArtDataset(trainingdir)
testingdataset = WikiArtDataset(testingdir)

print(traindataset.imgdir)

the_image, the_label = traindataset[5]
print(the_image, the_image.size())

# the_showable_image = F.to_pil_image(the_image)
# print("Label of img 5 is {}".format(the_label))
# the_showable_image.show()

class WikiArtModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        self.conv2d = nn.Conv2d(3, 3, (5,5), padding=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3*416*416, 300)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(300, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d(image)
        output = self.flatten(output)
        output = self.linear1(output)
        output = self.tanh(output)
        output = self.linear2(output)
        return self.softmax(output)

def train(epochs=3, modelfile=None):
    loader = DataLoader(traindataset, batch_size=32, shuffle=True)

    model = WikiArtModel()
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, y = batch
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output)
            loss.backward()
            optimizer.step()

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

model = train()
