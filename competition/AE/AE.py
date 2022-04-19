from pandas._libs.lib import indices_fast
import torch
from torch.utils.data import dataloader
from competition.utils import utils
from competition.utils.CustomDataset import CustomImageDataset
import torchvision as tv
from torch.nn import functional as F
from time import time
from tqdm import tqdm

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
        "AE", help="Train and Test the AE model")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class AE(torch.nn.Module):
    def __init__(self,train=True):
        super().__init__()

        self.is_training = train

        self.conv1 = torch.nn.Conv2d(1, 5, 5)
        self.pool1 = torch.nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = torch.nn.Conv2d(5, 10, 5)
        self.pool2 = torch.nn.MaxPool2d(2, 2, return_indices=True)
        self.conv3 = torch.nn.Conv2d(10, 20, 3)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(80, 40)
        self.fc2 = torch.nn.Linear(40, 20)
        self.fc3 = torch.nn.Linear(20, 10)

        self.fc4 = torch.nn.Linear(10, 20)
        self.fc5 = torch.nn.Linear(20, 40)
        self.fc6 = torch.nn.Linear(40, 80)
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(20, 2, 2))
        self.conv4 = torch.nn.ConvTranspose2d(20, 10, 3)
        self.pool3 = torch.nn.MaxUnpool2d(2, 2)
        self.conv5 = torch.nn.ConvTranspose2d(10, 5, 5)
        self.pool4 = torch.nn.MaxUnpool2d(2, 2)
        self.conv6 = torch.nn.ConvTranspose2d(5, 1, 5)

    def forward(self, x):
        x, indices1 = self.pool1(F.relu(self.conv1(x)))
        x, indices2 = self.pool2(F.relu(self.conv2(x)))

        x = self.conv3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        if(not(self.is_training)):
            return x

        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.unflatten(x)
        x = self.pool3(F.relu(self.conv4(x)), indices2)
        x = self.pool4(F.relu(self.conv5(x)), indices1)
        x = self.conv6(x)
        return x


def train():

    train_path = utils.get_path("../../datasets/mnist/training/")
    train_ann = utils.get_path("../../datasets/mnist/training/labels_train.csv")

    training_data = CustomImageDataset(
        annotations_file=train_ann,
        img_dir=train_path,
        transform=tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor()
        ])
    )

    train_loder = dataloader.DataLoader(
        training_data, batch_size=32, shuffle=True)


    epochs = 1

    loss_function = torch.nn.MSELoss()

    model = AE()
    lr = 0.001

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-8)

    device = torch.device("cpu")

    model.to(device=device)

    model.train() # missing function?

    running_loss = 0

    for i in range(epochs):
        for  data in tqdm(train_loder,desc=f"{i} Epoch: ",ascii="         #"):
            image, _ = data
            optimizer.zero_grad()

            output = model(image)
            loss = loss_function(image, output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Loss at {i+1} Epoch: {running_loss/len(train_loder)}")
    path_model = utils.get_path(
        f"../../models/AE/AE-{str(int(time()))}.pth")
    torch.save(model.state_dict(), path_model)

    return path_model


def evaluate(path_model):

    # re-think this function more clearly

    test_ann = utils.get_path(
        "../../datasets/mnist/validation/query/labels_query.csv")
    test_path = utils.get_path("../../datasets/mnist/validation/query")

    test_data = CustomImageDataset(
        annotations_file=test_ann,
        img_dir=test_path,
        transform=tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor()
        ])
    )

    test_dataloader = dataloader.DataLoader(
        test_data, batch_size=9, shuffle=True)
    dataiter = iter(test_dataloader)
    images, _ = dataiter.next()

    model = AE()
    model.load_state_dict(torch.load(path_model))

    outputs = model(images)

    for output in outputs[1:]:
        difference = outputs[0] - output
        print(torch.norm(difference).item())
    utils.imshow(tv.utils.make_grid(images))



def main(args):
    utils.createLabelsCsv("../../datasets/mnist/training/", "labels_train.csv")
    utils.createLabelsCsv(
        "../../datasets/mnist/validation/query", "labels_query.csv")

    if(args.test != None):
        path_model = utils.get_path(
            f"../../models/AE/AE-{args.test}.pth")
    else:
        path_model = train()

    evaluate(path_model)
