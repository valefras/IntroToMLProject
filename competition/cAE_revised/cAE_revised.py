from PIL.Image import Image
import matplotlib
from matplotlib.pyplot import cla, imshow
from matplotlib.transforms import Transform
import numpy
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
from tqdm import tqdm
from torch.utils.data import dataloader
import pandas as pd
from pandas import DataFrame
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
        "cAE_revised", help="Train and Test the AE with revised topology model (Animals dataset)")
    parser.add_argument(
        "--test", action='store', help="Only test the model with the given weight file", type=str
    )

    parser.set_defaults(func=main)


class cAE(torch.nn.Module):
    def __init__(self,train=True):
        super().__init__()

        self.is_training = train

        self.conv1 = torch.nn.Conv2d(1, 32, 5,stride=2) # -> 110x110x32
        self.conv2 = torch.nn.Conv2d(32, 64, 3,stride=2,padding=2) # -> 56x56x64
        self.conv4 = torch.nn.Conv2d(64,128,3) # -> 25x25x128
        self.conv5 = torch.nn.Conv2d(128,128,3,stride=3) # -> 8x8x128
        self.avg = torch.nn.AvgPool2d(8) # -> 1x1x128
        self.flat = torch.nn.Flatten()
        self.drop = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(128,128)


        self.classLayer = torch.nn.Linear(128,95)

        self.fc2 = torch.nn.Linear(128,128) # -> 128
        self.unflat = torch.nn.Unflatten(dim=1,unflattened_size=(128,1,1)) # -> 1x1x128
        self.avg2 = torch.nn.AdaptiveAvgPool2d((8,8)) # -> 8x8x128
        self.conv6 = torch.nn.ConvTranspose2d(128,128,3,stride=3,output_padding=(1,1)) # -> 25x25x128
        self.conv7 = torch.nn.ConvTranspose2d(128,64,3,dilation=2,padding=1,output_padding=(1,1)) # -> 28x28x128
        self.conv9 = torch.nn.ConvTranspose2d(64,32,3,stride=2,padding=2,output_padding=(1,1)) # -> 110x110x32
        self.conv10 = torch.nn.ConvTranspose2d(32,1,5,stride=2,output_padding=(1,1)) # -> 224x244x3

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avg(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc1(x)
        features = x

        classes = self.classLayer(features)

        x = self.fc2(x)
        x = self.unflat(x)
        x = self.avg2(x)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        return x, features,classes
class Competition_AE(CompetitionModel):
    def __init__(self, model, optim, loss, transform, name, dataset, epochs,channels=3):
        super().__init__(model, optim, loss, transform, name, dataset, epochs,channels)
        self.acc_classes = {}

    def computeLoss(self,inputs,outputs,labels,truthLabels):
        return self.loss_f(inputs,outputs)+torch.nn.CrossEntropyLoss()(labels,truthLabels)
    def calc_similarity(self, feats1, feats2):
        return numpy.linalg.norm(feats1-feats2)
        # return numpy.dot(feats1,feats2) / (numpy.linalg.norm(feats1) * numpy.linalg.norm(feats2))
    def fitModel(self,data,isTraining):
        image, labels, img_path = data

        if(not(isTraining)):
            with torch.no_grad():
                output = self.model(image)
                loss = self.computeLoss(image, output[0], output[2],labels)

        else:
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.computeLoss(image, output[0],output[2] ,labels)
        return loss
    def evaluate(self, path_model):
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
            test_data, batch_size=1, shuffle=True)

        self.model.load_state_dict(torch.load(path_model))

        self.model.eval()
        print("Evaluating data")
        feats_gallery = self.scan_gallery(path_model)

        for data in tqdm(test_dataloader, desc="Comparing gallery to query", ascii=" >>>>>>>>="):
            image, label, file_path = data
            with torch.no_grad():
                image_rec,features,classes_predicted = self.model(image)
                class_predicted = torch.argmax(classes_predicted)
                res = self.get_top10(features,class_predicted.item(), feats_gallery)
                self.get_score(res, label.item())
            # images = [Image.open(im) for im in res['path'].head(10)]
            # images.insert(0, Image.open(file_path[0]))
            # utils.display_images(images)
            # print(res)

        for key in self.score:
            print(key)
            print((self.score[key]*100) / len(test_dataloader))

        print(self.score)
    def get_top10(self, query_features,class_predicted, df_gallery: DataFrame):
        top10 = pd.DataFrame(columns=['label', 'distance', 'path'])
        for i, im in df_gallery.iterrows():
            sim = self.calc_similarity(query_features, im['features'])
            if(class_predicted != im['label']):
                sim = sim + 0.3*sim
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



def main(args):
    loss_function = torch.nn.MSELoss()

    net = cAE()
    lr = 0.0001

    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-6,)

    model_transform = tv.transforms.Compose([
        tv.transforms.Resize((112, 112)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        tv.transforms.Grayscale(),
    ])

    model = Competition_AE(net,optimizer,loss_function,model_transform,"cAE_revised","new_animals",50,channels=3)

    if(args.test != None):
        if args.test == "latest":
            path_model = utils.get_latest_model("cAE_revised")
        else:
            path_model = utils.get_path(f"../../models/cAE_revised/cAE_revised-{args.test}.pth")
    else:
        path_model = model.train()
    model.evaluate(path_model)
