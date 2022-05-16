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
        "comp_classifier128", help="Train and Test the comp_classifier128 model")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class comp_classifier128(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 128x128x3
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        # 126×126×16
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        # 124x124x32
        self.conv3 = torch.nn.Conv2d(32, 64, 2, stride=2)
        # 62x62x64
        self.conv4 = torch.nn.Conv2d(64, 128, 3)
        # 60x60x128
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        # 30x30x128
        self.conv5 = torch.nn.Conv2d(128, 256, 3, 2)
        # 14x14x256
        self.conv6 = torch.nn.Conv2d(256, 512, 3, 2)
        # 6x6x512
        self.avg = torch.nn.AvgPool2d(5)
        self.flatten = torch.nn.Flatten()
        self.drop = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.avg(x)
        x = self.flatten(x)
        x = self.drop(x)
        features = x
        x = F.relu(self.fc1(x))
        return x, features


class Competition_classifier(CompetitionModel):
    def __init__(self, model, optim, loss, transform, test_transform, name, dataset, epochs):
        super().__init__(model, optim, loss, transform,
                         test_transform, name, dataset, epochs)

    def computeLoss(self, inputs, outputs, labels):
        return self.loss_f(outputs, labels)


def main(args):
    loss_function = torch.nn.CrossEntropyLoss()

    net = comp_classifier128()
    lr = 0.001

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-8)

    model_transform = tv.transforms.Compose([
        tv.transforms.Resize((128, 128)),
        tv.transforms.ColorJitter(hue=.05, saturation=.05),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),

    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Resize((128, 128)),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),

    ])

    model = Competition_classifier(
        net, optimizer, loss_function, model_transform, test_transform, "comp_classifier128", "animal_scraped", 30)

    if(args.test != None):
        if args.test == "latest":
            path_model = utils.get_latest_model("comp_classifier128")
        else:
            path_model = utils.get_path(
                f"../../models/comp_classifier128/comp_classifier128-{args.test}.pth")
    else:
        path_model = model.train()
    model.evaluate(path_model)
