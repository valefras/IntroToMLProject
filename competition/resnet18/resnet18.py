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
        "resnet18", help="Train and Test the resnet18 model")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class resnet18(CompetitionModel):
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = tv.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    pretrained = True

    model = resnet18(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained),
            None,None,model_transform,test_transform,"resnet18","scraped_fixed",100,True,pretrained)

    if(args.test != None):
        if args.test == "latest":
            path_model = utils.get_latest_model("resnet18")
        else:
            path_model = utils.get_path(
                f"../../models/resnet18/resnet18-{args.test}.pth")
    else:
        path_model = model.train()
    model.evaluate(path_model)