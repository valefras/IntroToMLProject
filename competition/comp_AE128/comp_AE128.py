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
        "comp_AE128", help="Train and Test the AE model (Animals dataset)")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class comp_AE128(torch.nn.Module):
    def __init__(self, train=True):
        super().__init__()

        self.is_training = train
        # 128x128x3
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        # 124×124×16
        self.pool1 = torch.nn.MaxPool2d(2, 2, return_indices=True)
        # 62x62x16
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        # 58x58x32
        self.pool2 = torch.nn.MaxPool2d(2, 2, return_indices=True)
        # 29x29x32
        self.conv3 = torch.nn.Conv2d(32, 64, 5)
        # 25x25x64
        self.conv4 = torch.nn.Conv2d(64, 128, 6)
        # 20x20x128
        self.pool3 = torch.nn.MaxPool2d(2, 2, return_indices=True)
        # 10x10x128
        self.conv5 = torch.nn.Conv2d(128, 256, 5)
        # 6x6x256
        self.pool4 = torch.nn.MaxPool2d(2, 2, return_indices=True)
        # 3x3x256
        self.conv6 = torch.nn.Conv2d(256, 512, 3)
        # 1x1x512

        self.flatten = torch.nn.Flatten()
        # self.fc1 = torch.nn.Linear(2304, 1000)
        # self.fc2 = torch.nn.Linear(1000, 500)
        # # self.fc3 = torch.nn.Linear(5000, 1000)
        # # self.fc4 = torch.nn.Linear(500, 250)
        # # self.fc5 = torch.nn.Linear(250, 100)
        # # self.fc6 = torch.nn.Linear(100, 50)

        # # self.fc7 = torch.nn.Linear(50, 100)
        # # self.fc8 = torch.nn.Linear(100, 250)
        # # self.fc9 = torch.nn.Linear(250, 500)
        # # self.fc10 = torch.nn.Linear(1000, 5000)
        # self.fc11 = torch.nn.Linear(500, 1000)
        # self.fc12 = torch.nn.Linear(1000, 2304)

        self.unflatten = torch.nn.Unflatten(
            dim=1, unflattened_size=(512, 1, 1))
        self.conv7 = torch.nn.ConvTranspose2d(512, 256, 3)
        self.pool5 = torch.nn.MaxUnpool2d(2, 2)
        self.conv8 = torch.nn.ConvTranspose2d(256, 128, 5)
        self.pool6 = torch.nn.MaxUnpool2d(2, 2)
        self.conv9 = torch.nn.ConvTranspose2d(128, 64, 6)
        self.conv10 = torch.nn.ConvTranspose2d(64, 32, 5)
        self.pool7 = torch.nn.MaxUnpool2d(2, 2)
        self.conv11 = torch.nn.ConvTranspose2d(32, 16, 5)
        self.pool8 = torch.nn.MaxUnpool2d(2, 2)
        self.conv12 = torch.nn.ConvTranspose2d(16, 3, 5)

    def forward(self, x):
        x, indices1 = self.pool1(F.relu(self.conv1(x)))
        x, indices2 = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        x, indices3 = self.pool3(F.relu(self.conv4(x)))

        x, indices4 = self.pool4(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))

        x = self.flatten(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))

        features = x

        # x = F.relu(self.fc7(x))
        # x = F.relu(self.fc8(x))
        # x = F.relu(self.fc9(x))
        #x = F.relu(self.fc10(x))
        # x = F.relu(self.fc11(x))
        # x = F.relu(self.fc12(x))
        x = self.unflatten(x)
        x = self.pool5(F.relu(self.conv7(x)), indices4)
        x = self.pool6(F.relu(self.conv8(x)), indices3)
        x = F.relu(self.conv9(x))
        x = self.pool7(F.relu(self.conv10(x)), indices2)
        x = self.pool8(F.relu(self.conv11(x)), indices1)
        x = F.relu(self.conv12(x))
        return x, features


class Competition_AE(CompetitionModel):
    def __init__(self, model, optim, loss, transform, name, dataset, epochs, channels=3):
        super().__init__(model, optim, loss, transform, name, dataset, epochs, channels)

    def computeLoss(self, inputs, outputs, labels):
        return self.loss_f(inputs, outputs)


def main(args):
    loss_function = torch.nn.MSELoss()

    net = comp_AE128()
    lr = 0.001

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-8)

    model_transform = tv.transforms.Compose([
        tv.transforms.Resize((128, 128)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor()
    ])

    model = Competition_AE(net, optimizer, loss_function,
                           model_transform, "comp_AE128", "new_animals", 20)

    if(args.test != None):
        path_model = utils.get_path(
            f"../../models/comp_AE128/comp_AE128-{args.test}.pth")
    else:
        path_model = model.train()

    model.evaluate(path_model)
