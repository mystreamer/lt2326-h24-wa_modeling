import sys
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
# from torch.utils.data WeightedRandomSampler


class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            self.loaded = True

        return self.image

class WikiArtDataset(Dataset):
    def __init__(self, imgdir, device="cpu"):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        classes = set()
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            for art in artfiles:
                filedict[art] = WikiArtImage(imgdir, arttype, art)
                indices.append(art)
                classes.add(arttype)
        print("...finished")
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.classes = list(classes)
        self.device = device
 
    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)
        return image, ilabel

    def get_balancing_weights(self):
        """ Returns a list of tuples in the form
            (key, weight), where weight is the probability
            for a random weighted sampler that make the
            sampling probabilities balanced.
        """
        nclasses = len(self.classes)
        nimages = len(self.indices)
        count_per_class = {k: v for k, v in zip(self.classes, [0] * nclasses)}
        print(count_per_class)
        # make histogram
        for key in self.indices:
            count_per_class[self.filedict[key].label] += 1
        weight_per_class = {k: v for k, v in zip(self.classes, [0.] * nclasses)}
        print(count_per_class)
        for key in self.classes:
            weight_per_class[key] = float(nimages) / float(count_per_class[key])
        print(weight_per_class)
        weights = []
        # return per-datapoint weightings
        for _, idx in enumerate(self.indices):
            weights.append(weight_per_class.get(self.filedict[idx].label, 0))
        return weights

class WikiArtModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        self.conv2d = nn.Conv2d(3, 1, (4,4), padding=2)
        self.maxpool2d = nn.MaxPool2d((4,4), padding=2)
        self.flatten = nn.Flatten()
        self.batchnorm1d = nn.BatchNorm1d(105*105)
        self.linear1 = nn.Linear(105*105, 300)
        self.dropout = nn.Dropout(0.01)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(300, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d(image)
        #print("convout {}".format(output.size()))
        output = self.maxpool2d(output)
        #print("poolout {}".format(output.size()))        
        output = self.flatten(output)
        output = self.batchnorm1d(output)
        #print("poolout {}".format(output.size()))        
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        return self.softmax(output)
