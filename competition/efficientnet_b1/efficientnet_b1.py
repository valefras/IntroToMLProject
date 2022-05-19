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
        "efficientnet_b1", help="Train and Test the efficientnet_b1 model")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class efficientnet_b1(CompetitionModel):
    def __init__(self, model, optim, loss, transform, test_transform, name, dataset, epochs, premade, pretrained):
        super().__init__(model, optim, loss, transform,
                         test_transform, name, dataset, epochs, premade, pretrained)

    def computeLoss(self, inputs, outputs, labels):
        return self.loss_f(outputs, labels)


def main(args):
    model_transform = tv.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    test_transform = tv.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    pretrained = True

    lr = 0.001

    loss_function = torch.nn.CrossEntropyLoss()

    net = tv.models.efficientnet_b1(pretrained=True)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-8)

    model = efficientnet_b1(net,
                      optimizer, loss_function, model_transform, test_transform, "efficientnet_b1", "scraped_fixed", 100, True, pretrained)

    if(args.test != None):
        if args.test == "latest":
            path_model = utils.get_latest_model("efficientnet_b1")
        else:
            path_model = utils.get_path(
                f"../../models/efficientnet_b1/efficientnet_b1-{args.test}.pth")
    else:
        path_model = model.train()
    model.evaluate(path_model)
