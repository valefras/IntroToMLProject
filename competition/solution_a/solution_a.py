import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import DataLoader
from sys import path
from ..solution_a import utils


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def configure_subparsers(subparsers):
    r"""Configure a new subparser for our first solution of the competition.
    Args:
      subparsers: subparser
    """

    """
    Subparser parameters:
    Args:
    """
    parser = subparsers.add_parser(
        "solution_a", help="Train the Solution A model")
    parser.add_argument(
        "--test", action='store_true', default=False, help="Test the model"
    )
    parser.add_argument(
        "--generate_dataset", action="store_true", default=False, help="Generate train and test data MOCKUP"
    )

    parser.set_defaults(func=main)


def generate_dataset():
    utils.createLabelsCsv("../../datasets/mnist/training/", "labels_train.csv")
    utils.createLabelsCsv(
        "../../datasets/mnist/validation/", "labels_test.csv")
    train_path = utils.get_path("../../datasets/mnist/training/")
    train_ann = utils.get_path("./labels_train.csv")
    test_ann = utils.get_path("./labels_test.csv")
    test_path = utils.get_path("../../datasets/mnist/validation/")

    training_data = CustomImageDataset(
        annotations_file=train_ann,
        img_dir=train_path,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
    )

    test_data = CustomImageDataset(
        annotations_file=test_ann,
        img_dir=test_path,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


def main(args):
    print("Main function of Solution A")

    print("Parameters given:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))

    if args.generate_dataset:
        generate_dataset()

    print(args.test)
