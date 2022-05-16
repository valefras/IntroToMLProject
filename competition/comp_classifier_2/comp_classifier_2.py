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
from torch.optim.lr_scheduler import ExponentialLR
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
        "comp_classifier_2", help="Train and Test the comp_classifier model")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class comp_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # channels 3, filter count 32, spatial extent 5
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 94, 3, stride=2)
        self.conv4 = torch.nn.Conv2d(94, 94, 3)
        self.avg = torch.nn.AvgPool2d(7)
        self.flatten = torch.nn.Flatten()
        self.drop = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(94, 94)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
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
        #self.ELR = ExponentialLR(optimizer=self.optimizer,gamma=0.95, verbose = True)

    def computeLoss(self, inputs, outputs, labels):
        return self.loss_f(outputs, labels)


def main(args):
    loss_function = torch.nn.CrossEntropyLoss()

    net = comp_classifier()
    lr = 0.01

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-6)

    model_transform = tv.transforms.Compose([
        tv.transforms.Resize((84, 84)),
        tv.transforms.ColorJitter(hue=.05, saturation=.05),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                0.229, 0.224, 0.225])
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Resize((84, 84)),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                0.229, 0.224, 0.225])
    ])

    model = Competition_classifier(
        net, optimizer, loss_function, model_transform, test_transform, "comp_classifier_2", "scraped_fixed", 100)

    if(args.test != None):
        if args.test == "latest":
            path_model = utils.get_latest_model("comp_classifier_2")
        else:
            path_model = utils.get_path(
                f"../../models/comp_classifier_2/comp_classifier_2-{args.test}.pth")
    else:
        path_model = model.train()
    model.evaluate(path_model)
