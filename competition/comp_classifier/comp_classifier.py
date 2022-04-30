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
        "comp_classifier", help="Train and Test the comp_classifier model")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class comp_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 5)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(5, 10, 5)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(10, 15, 3)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(1815, 900)
        self.fc2 = torch.nn.Linear(900, 200)
        self.fc3 = torch.nn.Linear(200, 100)
        self.fc4 = torch.nn.Linear(100, 30)
        self.fc5 = torch.nn.Linear(30, 9)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        features = x
        x = F.relu(self.fc5(x))
        return x, features


class Competition_classifier(CompetitionModel):
    def __init__(self, model, optim, loss, transform, name, dataset, epochs):
        super().__init__(model, optim, loss, transform, name, dataset, epochs)

    def computeLoss(self, inputs, outputs, labels):
        return self.loss_f(outputs, labels)


def main(args):
    loss_function = torch.nn.CrossEntropyLoss()

    net = comp_classifier()
    lr = 0.001

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-8)

    model_transform = tv.transforms.Compose([
        tv.transforms.Resize((64, 64)),
        tv.transforms.ColorJitter(hue=.05, saturation=.05),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor()
    ])

    model = Competition_classifier(
        net, optimizer, loss_function, model_transform, "comp_classifier", "animals", 175)

    if(args.test != None):
        path_model = utils.get_path(
            f"../../models/comp_classifier/comp_classifier-{args.test}.pth")
    else:
        path_model = model.train()

    model.evaluate(path_model)
