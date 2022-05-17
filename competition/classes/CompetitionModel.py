from abc import abstractmethod
import numpy
from pandas.core.frame import DataFrame
from scipy.spatial import distance
from numpy.linalg import inv
import torch
from torch.utils.data import dataloader, random_split
from torchvision import transforms
from competition.utils import utils
from competition.utils.CustomDataset import CustomImageDataset
import torchvision as tv
from torch.nn import functional as F
from time import time
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
import abc
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class CompetitionModel():
    def __init__(self, model, optim, loss, transform, test_transform, name, dataset, epochs, resnet=False, pretrained=False, channels=3):
        self.model = model
        self.optimizer = optim
        self.loss_f = loss
        self.name = name
        self.dataset = dataset
        self.transform = transform
        self.test_transform = test_transform
        self.epochs = epochs
        self.n_channels = channels
        self.resnet = resnet
        self.pretrained = pretrained

        self.score = {
            'top1': 0,
            'top5': 0,
            'top10': 0
        }

    def scan_gallery(self, path_model=None):
        utils.createLabelsCsv(
            f"../../datasets/{self.dataset}/validation/gallery", "labels_gallery.csv")
        gallery_path = utils.get_path(
            f"../../datasets/{self.dataset}/validation/gallery")
        gallery_ann = utils.get_path(
            f"../../datasets/{self.dataset}/validation/gallery/labels_gallery.csv")
        gallery_data = CustomImageDataset(
            annotations_file=gallery_ann,
            img_dir=gallery_path,
            transform=self.test_transform
        )

        gallery_dataloader = dataloader.DataLoader(
            gallery_data, batch_size=1, shuffle=False)

        if not(self.resnet and self.pretrained):
            self.model.load_state_dict(torch.load(path_model))

        features_gallery = pd.DataFrame(columns=['label', 'features', 'path'])

        for data in tqdm(gallery_dataloader, desc="Extracting features from the gallery", ascii=" >>>>>>>>="):
            image, label, file_path = data
            with torch.no_grad():
                if not self.resnet:
                    features_gallery = features_gallery.append(pd.DataFrame({'label': label.item(), 'features': [
                        self.model(image)[1].numpy()[0]], 'path': file_path[0]}), ignore_index=False)
                else:
                    features_gallery = features_gallery.append(pd.DataFrame({'label': label.item(), 'features': [
                        self.model(image).numpy()[0]], 'path': file_path[0]}), ignore_index=False)

        return features_gallery

    def calc_similarity(self, feats1, feats2):
        # return numpy.linalg.norm(feats1 - feats2)**2 # Euclidean
        return distance.euclidean(feats1, feats2)

        # cosine similarity
        # return (numpy.dot(feats1, feats2) / (numpy.linalg.norm(feats1)*numpy.linalg.norm(feats2)))

        # mahalanobis distance, NOT TESTED
        """ X = numpy.stack((feats1, feats2), axis=0)
        iv = inv(numpy.cov(X))
        return distance.mahalanobis(feats1, feats2, iv) """

    def get_top10(self, query_image, df_gallery: DataFrame):
        top10 = pd.DataFrame(columns=['label', 'distance', 'path'])
        for i, im in df_gallery.iterrows():
            sim = self.calc_similarity(query_image, im['features'])
            if(len(top10.index) < 10):
                top10 = top10.append(pd.DataFrame({'label': [im['label']], 'distance': [
                                     sim], 'path': [im['path']]}), ignore_index=False)
            else:
                if(top10['distance'].iloc[-1] > sim):
                    top10 = top10.head(-1)
                    top10 = top10.append(pd.DataFrame({'label': [im['label']], 'distance': [
                                         sim], 'path': [im['path']]}), ignore_index=False)
            top10 = top10.sort_values(by=['distance'], ascending=True)
        return top10

    def find_image(self, gallery_path, query, file_name, path_model):
        pass

    def train(self):
        utils.createLabelsCsv(
            f"../../datasets/{self.dataset}/training/", "labels_train.csv")
        utils.createLabelsCsv(
            f"../../datasets/{self.dataset}/validation/query", "labels_query.csv")
        dataset_path = utils.get_path(
            f"../../datasets/{self.dataset}/training/")
        dataset_ann = utils.get_path(
            f"../../datasets/{self.dataset}/training/labels_train.csv")

        dataset = CustomImageDataset(
            annotations_file=dataset_ann,
            img_dir=dataset_path,
            transform=self.transform,
            channels=self.n_channels
        )
        train_split_size = int(len(dataset)*0.8)
        validation_split_size = len(dataset) - train_split_size

        train_split, validation_split = random_split(
            dataset, [train_split_size, validation_split_size])

        train_loder = dataloader.DataLoader(
            train_split, batch_size=32, shuffle=True)

        validation_loader = dataloader.DataLoader(
            validation_split, batch_size=32, shuffle=True)

        device = torch.device("cpu")

        self.model.to(device=device)

        min_val_error = float("inf")
        path_model = ""
        if not os.path.exists(utils.get_path(f"../../models/{self.name}/")):
            os.makedirs(utils.get_path(f"../../models/{self.name}/"))
        for i in range(self.epochs):
            self.model.train()
            for data in tqdm(train_loder, desc=f"{i+1} Epoch: ", ascii=" >>>>>>>>>="):

                loss = self.fitModel(data, True)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            running_loss = 0
            for data in tqdm(validation_loader, desc="Computing the model's validation error", ascii=" >>>>>>>>="):
                self.fitModel(data, False)
                running_loss += loss.item()

            print(
                f"Validation loss at {i+1} Epoch: {running_loss/len(validation_loader)}")
            # at least 1 time this condition will return true, so train() will always return a correct path.
            if(running_loss < min_val_error):
                print("Got better validation error, saving model's weight\n")
                min_val_error = running_loss
                path_model = self.save_weights()
                if(self.ELR != None):
                    self.ELR.step()
        return path_model

    def fitModel(self, data, isTraining):

        image, labels, img_path = data

        if(not(isTraining)):
            with torch.no_grad():
                output = self.model(image)

        else:
            self.optimizer.zero_grad()
            output = self.model(image)
        if self.resnet:
            return self.computeLoss(image, output, labels)

        return self.computeLoss(image, output[0], labels)

    def evaluate(self, path_model=None):
        test_ann = utils.get_path(
            f"../../datasets/{self.dataset}/validation/query/labels_query.csv")
        test_path = utils.get_path(
            f"../../datasets/{self.dataset}/validation/query")

        test_data = CustomImageDataset(
            annotations_file=test_ann,
            img_dir=test_path,
            transform=self.test_transform

        )

        test_dataloader = dataloader.DataLoader(
            test_data, batch_size=1, shuffle=True)

        if not(self.resnet and self.pretrained):
            self.model.load_state_dict(torch.load(path_model))

        self.model.eval()
        print("Evaluating data")
        feats_gallery = self.scan_gallery(path_model)

        # for data in tqdm(test_dataloader, desc="Comparing gallery to query", ascii=" >>>>>>>>="):
        #     image, label, file_path = data
        #     with torch.no_grad():
        #         res = self.get_top10(self.model(
        #             image)[1].numpy()[0], feats_gallery)
        #         self.get_score(res, label.item())

        for data in tqdm(test_dataloader, desc="Comparing gallery to query", ascii=" >>>>>>>>="):
            image, label, file_path = data
            with torch.no_grad():
                res = self.model(image)
                if not self.resnet:
                    feats = res[1].numpy()[0]
                else:
                    feats = res.numpy()[0]
                # utils.imshow(res[0])
                res = self.get_top10(feats, feats_gallery)
                self.get_score(res, label.item())
                '''
                images = [Image.open(im) for im in res['path'].head(10)]
                images.insert(0, Image.open(file_path[0]))
                utils.display_images(images)
                '''

        for key in self.score:
            print(key)
            print((self.score[key]*100) / len(test_dataloader))

        print(self.score)

    def get_score(self, top10, query_label):
        # check top 1
        top10_vals = top10.label.values
        if(top10_vals[0] == query_label):
            self.score['top1'] += 1
            #self.score['top5'] += 1
            #self.score['top10'] += 1
        # check top 5
        if(query_label in top10_vals[:5]):
            self.score['top5'] += 1
            #self.score['top10'] += 1

        # check top 10
        if(query_label in top10_vals):
            self.score['top10'] += 1

    def save_weights(self):
        file_name = str(int(time()))

        path_model = utils.get_path(
            f"../../models/{self.name}/{self.name}-{file_name}.pth")
        torch.save(self.model.state_dict(), path_model)
        return path_model

    @abc.abstractmethod
    def computeLoss(self, inputs, outputs, labels):
        return self.loss_f(inputs, outputs, labels)
