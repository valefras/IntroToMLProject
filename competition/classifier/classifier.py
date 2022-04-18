from pandas._libs.lib import indices_fast
import torch
from torch.utils.data import dataloader
import torchvision as tv
from torch.nn import functional as F
import torch.optim as optim
from competition.utils import utils
from competition.utils.CustomDataset import CustomImageDataset
from time import time


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


def train(classes):

    train_path = utils.get_path("../../datasets/mnist/training/")
    train_ann = utils.get_path(
        "../../datasets/mnist/training/labels_train.csv")

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

    model = classifier()

    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    device = torch.device("cpu")
    model.to(device=device)
    
    epochs = 1
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loder):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    path_model = utils.get_path(
        f"../../models/classifier/classifier-{str(int(time()))}.pth")
    torch.save(model.state_dict(), path_model)

    return path_model


def evaluate(path_model, classes):
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
    images, labels = dataiter.next()

    # print images
    utils.imshow(tv.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))


def main(args):
    utils.createLabelsCsv("../../datasets/mnist/training/", "labels_train.csv")
    labels = utils.createLabelsCsv(
        "../../datasets/mnist/validation/query", "labels_query.csv")

    if(args.test != None):
        path_model = utils.get_path(
            f"../../models/classifier/classifier-{args.test}.pth")
    else:
        path_model = train(labels)

    evaluate(path_model, labels)
