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
        "comp_AE", help="Train and Test the AE model (Animals dataset)")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class comp_AE(torch.nn.Module):
    def __init__(self, train=True):
        super().__init__()

        self.is_training = train

        self.conv1 = torch.nn.Conv2d(3, 5, 5)
        self.pool1 = torch.nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = torch.nn.Conv2d(5, 10, 5)
        self.pool2 = torch.nn.MaxPool2d(2, 2, return_indices=True)
        self.conv3 = torch.nn.Conv2d(10, 15, 5)
        self.conv4 = torch.nn.Conv2d(15, 15, 5)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(1500, 1000)
        self.fc2 = torch.nn.Linear(1000, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, 50)

        self.fc5 = torch.nn.Linear(50, 100)
        self.fc6 = torch.nn.Linear(100, 500)
        self.fc7 = torch.nn.Linear(500, 1000)
        self.fc8 = torch.nn.Linear(1000, 1500)
        self.unflatten = torch.nn.Unflatten(
            dim=1, unflattened_size=(15, 10, 10))
        self.conv5 = torch.nn.ConvTranspose2d(15, 15, 5)
        self.conv6 = torch.nn.ConvTranspose2d(15, 10, 5)
        self.pool3 = torch.nn.MaxUnpool2d(2, 2)
        self.conv7 = torch.nn.ConvTranspose2d(10, 5, 5)
        self.pool4 = torch.nn.MaxUnpool2d(2, 2)
        self.conv8 = torch.nn.ConvTranspose2d(5, 3, 5)

    def forward(self, x):
        x, indices1 = self.pool1(F.relu(self.conv1(x)))
        x, indices2 = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        features = x

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.unflatten(x)
        x = F.relu(self.conv5(x))
        x = self.pool3(F.relu(self.conv6(x)), indices2)
        x = self.pool4(F.relu(self.conv7(x)), indices1)
        x = F.relu(self.conv8(x))
        return x, features


class Competition_AE(CompetitionModel):
    def __init__(self, model, optim, loss, transform, name, dataset, epochs, channels=3):
        super().__init__(model, optim, loss, transform, name, dataset, epochs, channels)

    def computeLoss(self, inputs, outputs, labels):
        return self.loss_f(inputs, outputs)


def main(args):
    loss_function = torch.nn.MSELoss()

    net = comp_AE()
    lr = 0.001

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-8)

    model_transform = tv.transforms.Compose([
        tv.transforms.Resize((84, 84)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor()
    ])

    model = Competition_AE(net, optimizer, loss_function,
                           model_transform, "comp_AE", "new_animals", 1)

    if(args.test != None):
        if args.test == "latest":
            path_model = utils.get_latest_model("comp_AE")
        else:
            path_model = utils.get_path(f"../../models/comp_AE/comp_AE-{args.test}.pth")
    else:
        path_model = model.train()
    model.evaluate(path_model)