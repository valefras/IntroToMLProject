from PIL.Image import Image
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import cla, imshow
from matplotlib.transforms import Transform
import numpy
import torch
from torch.optim.adam import Adam
from torchvision import transforms
from competition.classes.CompetitionModel import CompetitionModel
from competition.utils import utils
from competition.utils.CustomDataset import CustomImageDataset
import torchvision as tv
from torch.nn import functional as F
import PIL
import warnings
from tqdm import tqdm
from torch.utils.data import dataloader
import pandas as pd
from pandas import DataFrame
warnings.simplefilter(action='ignore', category=UserWarning)
from torch.optim.lr_scheduler import ExponentialLR
from torchsummary import summary

from random import sample
def configure_subparsers(subparsers):
    r"""Configure a new subparser for our second solution of the competition.
    Args:
      subparsers: subparser
    """

    """
  Subparser parameters:
  Args:
  """
    parser = subparsers.add_parser(
        "cAE_revised", help="Train and Test the AE with revised topology model (Animals dataset)")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class cAE(torch.nn.Module):
    def __init__(self, train=True):
        super().__init__()

        self.is_training = train
        self.conv1 = torch.nn.Conv2d(1, 64, 3, stride=2)  # -> #55x55
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(
            64, 128, 3, stride=2, padding=2)  # -> 29x29
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, 3)  # -> 27x27x128
        self.bn3_4 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.Conv2d(256, 256, 3, stride=2)  # -> 13x13x128

        self.avg = torch.nn.AvgPool2d(13)  # -> 1x1x128
        self.flat = torch.nn.Flatten()
        self.drop = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(256, 256)

        self.classLayer = torch.nn.Linear(256, 94)


        self.fc2 = torch.nn.Linear(256, 256)  # -> 128
        self.unflat = torch.nn.Unflatten(
            dim=1, unflattened_size=(256, 1, 1))  # -> 1x1x128
        self.avg2 = torch.nn.AdaptiveAvgPool2d((13, 13))  # -> 8x8x128
        self.conv5 = torch.nn.ConvTranspose2d(
                256, 256, 3, stride=2)  # -> 25x25x128
        self.conv6 = torch.nn.ConvTranspose2d(
            256, 128, 3,stride=1)  # -> 28x28x128
        self.conv7 = torch.nn.ConvTranspose2d(
            128,64, 3, stride=2, padding=2)  # -> 110x110x32
        self.conv8 = torch.nn.ConvTranspose2d(
            64, 1, 5, stride=2,padding=1,output_padding=(1,1)) # -> 224x244x3

#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 64, 55, 55]             640 x -> x1
#             Conv2d-2          [-1, 128, 29, 29]          73,856 x1 -> x2 + x
#             Conv2d-3          [-1, 256, 27, 27]         295,168 x2 -> x3 + x1
#             Conv2d-4          [-1, 256, 13, 13]         590,080
#         #residual layers
        #128x28x28
        self.down1 = torch.nn.Sequential(
                torch.nn.Conv2d(1,128,1,stride=4,padding=1),
                torch.nn.BatchNorm2d(128)
                )
        # 256x26x26
        self.down2 = torch.nn.Sequential(
                torch.nn.Conv2d(128,256,4,stride=2),
                torch.nn.BatchNorm2d(256)
                )

    def forward(self, x):

        res1 = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) + self.down1(res1)
        res2 = x
        x = F.relu(self.bn3_4(self.conv3(x)))
        x = F.relu(self.bn3_4(self.conv4(x))) + self.down2(res2)
        x = self.avg(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc1(x)
        features = x

        classes = F.softmax(self.classLayer(features))

        x = F.relu(self.fc2(x))
        x = self.unflat(x)
        x = self.avg2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.sigmoid(self.conv8(x))



        # res1 = x
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x))) + self.down1(res1)
        # res2 = x
        # x = F.relu(self.bn3_4(self.conv3(x)))
        # x = F.relu(self.bn3_4(self.conv4(x))) + self.down2(res2)
        # x = self.avg(x)
        # x = self.flat(x)
        # x = self.drop(x)
        # x = self.fc1(x)
        # features = x
        #
        # classes = self.classLayer(features)
        #
        # x = self.fc2(x)
        # x = self.unflat(x)
        # x = self.avg2(x)
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))
        # x = F.relu(self.conv8(x))
        #
        return x, features, classes


class Competition_AE(CompetitionModel):
    def __init__(self, model, optim, loss, transform, test_transform, name, dataset, epochs, channels=3,premade=False):
        super().__init__(model, optim, loss, transform,
                         test_transform, name, dataset, epochs,premade,False, channels)
        self.acc_classes = {}
        self.ELR = ExponentialLR(optimizer=self.optimizer,gamma=0.95, verbose = True)

    def computeLoss(self, inputs, outputs, labels, truthLabels):
        return self.loss_f(inputs, outputs)+torch.nn.CrossEntropyLoss()(labels, truthLabels)

    def fitModel(self, data, isTraining):
        image, labels, img_path = data

        if(not(isTraining)):
            with torch.no_grad():
                output = self.model(image)
                loss = self.computeLoss(image, output[0], output[2], labels)

        else:
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.computeLoss(image, output[0], output[2], labels)
        return loss
    def fetch_outputs(self):

        test_ann = utils.get_path(
            f"../../datasets/{self.dataset}/validation/query/labels_query.csv")
        test_path = utils.get_path(
            f"../../datasets/{self.dataset}/validation/query")
        test_data = CustomImageDataset(
            annotations_file=test_ann,
            img_dir=test_path,
            transform=self.test_transform
        )

        test_dataloader = dataloader.DataLoader(
            test_data, batch_size=1, shuffle=True)


        for i,data in enumerate(tqdm(test_dataloader, desc="Saving reconstructed images", ascii=" >>>>>>>>=")):
            image,_,_ = data
            with torch.no_grad():
                image_rec, _ , _ = self.model(image)

                path = utils.get_path(f"../../reconstructions/{self.name}/{i}.png")


                fig,ax = plt.subplots(1,2)
                reconstruction = (image_rec.numpy()[0] + 1)/2

                ax[0].imshow(numpy.transpose(reconstruction,(1,2,0)))
                ax[1].imshow(image.squeeze(1).permute(1,2,0)/2)


                fig.savefig(path)
                plt.close(fig)



def main(args):
    loss_function = torch.nn.MSELoss()

    net = cAE()
    lr = 0.01

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-6,)

    model_transform = tv.transforms.Compose([
        tv.transforms.Resize((112, 112)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Grayscale(),
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Resize((112, 112)),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Grayscale(),
    ])

    print(summary(net,(1,112,112)))
    model = Competition_AE(net, optimizer, loss_function, model_transform, test_transform,
                           "cAE_revised", "scraped_fixed", 100,channels=3,premade=False)

    if(args.test != None):
        if args.test == "latest":
            path_model = utils.get_latest_model("cAE_revised")
        else:
            path_model = utils.get_path(
                f"../../models/cAE_revised/cAE_revised-{args.test}.pth")
    else:
        path_model = model.train()
    model.evaluate(path_model)
