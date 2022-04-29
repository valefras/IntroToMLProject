import torch
from torch.optim.adam import Adam
from torchvision import transforms
from competition.classes.CompetitionModel import CompetitionModel
from competition.utils import utils
from competition.utils.CustomDataset import CustomImageDataset
import torchvision as tv
from torch.nn import functional as F


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
        "classifier", help="Train and Test the classifier model")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(5, 10, 5)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(10, 20, 3)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(80, 40)
        self.fc2 = torch.nn.Linear(40, 20)
        self.fc3 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.conv3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Competition_classifier(CompetitionModel):
    def __init__(self, model, optim, loss, transform, name, dataset, epochs,n_channels):
        super().__init__(model, optim, loss, transform, name, dataset, epochs,n_channels)

    def computeLoss(self,inputs,outputs,labels):
        return self.loss_f(outputs, labels)

def main(args):

    utils.createLabelsCsv("../../datasets/mnist/training/", "labels_train.csv")
    utils.createLabelsCsv(
        "../../datasets/mnist/validation/query", "labels_query.csv")

    loss_function = torch.nn.CrossEntropyLoss()

    net = classifier()
    lr = 0.001

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-8)

    model_transform = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor()
    ])

    model = Competition_classifier(
        net, optimizer, loss_function, model_transform, "classifier", "mnist", 2,1)

    if(args.test != None):
        path_model = utils.get_path(f"../../models/classifier/classifier-{args.test}.pth")
    else:
        path_model = model.train()

    #model.evaluate(path_model)
