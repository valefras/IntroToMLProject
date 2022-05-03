from PIL.Image import Image
import matplotlib
from matplotlib.pyplot import imshow
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
warnings.simplefilter(action='ignore', category=UserWarning)

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
    def __init__(self,train=True):
        super().__init__()

        self.is_training = train

        self.conv1 = torch.nn.Conv2d(3, 32, 5,stride=2) # -> 110x110x32
        self.conv2 = torch.nn.Conv2d(32, 64, 3,stride=2,padding=2) # -> 56x56x64
        self.conv3 = torch.nn.Conv2d(64,128,3,stride=2,padding=1) # -> 28x28x128
        self.conv4 = torch.nn.Conv2d(128,128,3) # -> 25x25x128
        self.conv5 = torch.nn.Conv2d(128,128,3,stride=3) # -> 8x8x128
        self.avg = torch.nn.AvgPool2d(8) # -> 1x1x128
        self.flat = torch.nn.Flatten()
        self.drop = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(128,128)

        self.fc2 = torch.nn.Linear(128,128) # -> 128
        self.unflat = torch.nn.Unflatten(dim=1,unflattened_size=(128,1,1)) # -> 1x1x128
        self.avg2 = torch.nn.AdaptiveAvgPool2d((8,8)) # -> 8x8x128
        self.conv6 = torch.nn.ConvTranspose2d(128,128,3,stride=3,output_padding=(1,1)) # -> 25x25x128
        self.conv7 = torch.nn.ConvTranspose2d(128,128,3,dilation=2,padding=1,output_padding=(1,1)) # -> 28x28x128
        self.conv8 = torch.nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=(1,1)) # -> 56x56x64
        self.conv9 = torch.nn.ConvTranspose2d(64,32,3,stride=2,padding=2,output_padding=(1,1)) # -> 110x110x32
        self.conv10 = torch.nn.ConvTranspose2d(32,3,5,stride=2,output_padding=(1,1)) # -> 224x244x3

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avg(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc1(x)
        features = x

        x = self.fc2(x)
        x = self.unflat(x)
        x = self.avg2(x)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        return x, features
class Competition_AE(CompetitionModel):
    def __init__(self, model, optim, loss, transform, name, dataset, epochs,channels=3):
        super().__init__(model, optim, loss, transform, name, dataset, epochs,channels)

    def computeLoss(self,inputs,outputs,labels):
        return self.loss_f(inputs,outputs)
    def calc_similarity(self, feats1, feats2):
        return numpy.dot(feats1,feats2) / (numpy.linalg.norm(feats1) * numpy.linalg.norm(feats2))
def main(args):
    loss_function = torch.nn.MSELoss()

    net = cAE()
    lr = 0.001

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-8)

    model_transform = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor()
    ])

    model = Competition_AE(net,optimizer,loss_function,model_transform,"cAE_revised","new_animals",15)

    if(args.test != None):
        path_model = utils.get_path(f"../../models/cAE_revised/cAE_revised-{args.test}.pth")
    else:
        path_model = model.train()

    model.evaluate(path_model)

