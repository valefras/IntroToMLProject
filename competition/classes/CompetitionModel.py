from abc import abstractmethod
import numpy
from pandas._libs.lib import indices_fast
from pandas.core.frame import DataFrame
import torch
from torch.utils.data import dataloader
from torchvision import transforms
from competition.utils import utils
from competition.utils.CustomDataset import CustomImageDataset
import torchvision as tv
from torch.nn import functional as F
from time import time
from tqdm import tqdm
from PIL import Image
import os
import csv
import pprint

import pandas as pd
import abc


class CompetitionModel():
    def __init__(self, model, optim, loss, transform, name, dataset, epochs):
        self.model = model
        self.optimizer = optim
        self.loss_f = loss
        self.name = name
        self.dataset = dataset
        self.transform = transform
        self.epochs = epochs

    def scan_gallery(self, gallery_path, file_name, path_model):
        pass

    def calc_similarity(self, vector1, vector2):
        pass

    def get_top10(self, query_image, df_gallery: DataFrame):
        pass

    def find_image(self, gallery_path, query, file_name, path_model):
        pass

    def train(self):

        train_path = utils.get_path(f"../../datasets/{self.dataset}/training/")
        train_ann = utils.get_path(
            f"../../datasets/{self.dataset}/training/labels_train.csv")

        training_data = CustomImageDataset(
            annotations_file=train_ann,
            img_dir=train_path,
            transform=self.transform
        )

        train_loder = dataloader.DataLoader(
            training_data, batch_size=32, shuffle=True)

        device = torch.device("cpu")

        self.model.to(device=device)

        self.model.train()  # missing function?

        running_loss = 0

        for i in range(self.epochs):
            for data in tqdm(train_loder, desc=f"{i} Epoch: ", ascii=" >>>>>>>>>="):
                image, _ = data
                self.optimizer.zero_grad()

                output = self.model(image)
                loss = self.computeLoss(image, output, _)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(f"Loss at {i+1} Epoch: {running_loss/len(train_loder)}")

        file_name = str(int(time()))

        path_model = utils.get_path(
            f"../../models/{self.name}/AE-{file_name}.pth")
        torch.save(self.model.state_dict(), path_model)

        self.scan_gallery(
            f"../../datasets/{self.dataset}/validation/gallery/", file_name, path_model)

        self.find_image(f"../../datasets/{self.dataset}/validation/gallery/",
                        f"../../datasets/{self.dataset}/validation/query/3/5140.png", file_name, path_model)

        return path_model

    def evaluate(self, path_model):

        # re-think this function more clearly

        test_ann = utils.get_path(
            f"../../datasets/{self.dataset}/validation/query/labels_query.csv")
        test_path = utils.get_path(
            f"../../datasets/{self.dataset}/validation/query")

        test_data = CustomImageDataset(
            annotations_file=test_ann,
            img_dir=test_path,
            transform=self.transform
        )

        test_dataloader = dataloader.DataLoader(
            test_data, batch_size=9, shuffle=True)
        dataiter = iter(test_dataloader)
        images, _ = dataiter.next()

        self.model.load_state_dict(torch.load(path_model))

        outputs = self.model(images)

        for output in outputs[1:]:
            difference = outputs[0] - output
            print(torch.norm(difference).item())
        utils.imshow(tv.utils.make_grid(images))

    @abc.abstractmethod
    def computeLoss(self, inputs, outputs, labels):
        return  # self.loss_f(inputs,outputs,labels)
