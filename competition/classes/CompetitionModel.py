from abc import abstractmethod
import numpy
from pandas._libs.lib import indices_fast
from pandas.core.frame import DataFrame
import torch
from torch.utils.data import dataloader,random_split
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
    def __init__(self, model, optim, loss, transform, name, dataset, epochs,channels=3):
        self.model = model
        self.optimizer = optim
        self.loss_f = loss
        self.name = name
        self.dataset = dataset
        self.transform = transform
        self.epochs = epochs
        self.n_channels = channels
    def scan_gallery(self, gallery_path, file_name, path_model):
        pass

    def calc_similarity(self, vector1, vector2):
        pass

    def get_top10(self, query_image, df_gallery: DataFrame):
        pass

    def find_image(self, gallery_path, query, file_name, path_model):
        pass

    def train(self):

        dataset_path = utils.get_path(f"../../datasets/{self.dataset}/training/")
        dataset_ann = utils.get_path(
            f"../../datasets/{self.dataset}/training/labels_train.csv")
        dataset = CustomImageDataset(
            annotations_file=dataset_ann,
            img_dir=dataset_path,
            transform=self.transform,
            channels=self.n_channels
        )
        print(dataset_path)
        train_split_size = int(len(dataset)*0.8)
        validation_split_size = len(dataset) - train_split_size

        train_split, validation_split = random_split(dataset,[train_split_size,validation_split_size])


        train_loder = dataloader.DataLoader(
            train_split, batch_size=32, shuffle=True)

        validation_loader = dataloader.DataLoader(
            validation_split,batch_size=32,shuffle=False)

        device = torch.device("cpu")

        self.model.to(device=device)

        min_val_error = float("inf")
        path_model = ""
        for i in range(self.epochs):
            self.model.train()
            for data in tqdm(train_loder, desc=f"{i+1} Epoch: ", ascii=" >>>>>>>>>="):
                image, _ = data
                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.computeLoss(image, output, _)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            running_loss = 0
            for data in tqdm(validation_loader,desc="Computing the model's validation error", ascii=" >>>>>>>>="):
                image,_ = data
                with torch.no_grad():
                    output = self.model(image)
                loss = self.computeLoss(image,output,_)
                running_loss += loss.item()


            print(f"Validation loss at {i+1} Epoch: {running_loss/len(validation_loader)}")
            #at least 1 time this condition will return true, so train() will always return a correct path.
            if(running_loss < min_val_error):
                print("Got better validation error, saving model's weight\n")
                min_val_error = running_loss
                path_model = self.save_weights()

        # self.scan_gallery(
        #     f"../../datasets/{self.dataset}/validation/gallery/", file_name, path_model)
        #
        # self.find_image(f"../../datasets/{self.dataset}/validation/gallery/",
        #                 f"../../datasets/{self.dataset}/validation/query/3/5140.png", file_name, path_model)
        #
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

    def save_weights(self):
        file_name = str(int(time()))

        path_model = utils.get_path(
            f"../../models/{self.name}/{self.name}-{file_name}.pth")
        torch.save(self.model.state_dict(), path_model)
        return path_model


    @abc.abstractmethod
    def computeLoss(self, inputs, outputs, labels):
        return  # self.loss_f(inputs,outputs,labels)
