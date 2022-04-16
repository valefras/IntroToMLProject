import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def get_path(req_path):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), req_path))


def createLabelsCsv(ds_path, csv_name):
    path_set = get_path(ds_path)
    folder_walk = os.walk(path_set)
    labels = next(folder_walk)[1]

    slashes = "/"

    if(os.name == "nt"):
        # im on windows
        slashes = "\\"

    with open(os.path.realpath(os.path.join(path_set, csv_name)), "w") as f:
        csvwriter = csv.writer(f)
        for root, dirs, files in os.walk(path_set, topdown=True):
            for name in files:
                csvwriter.writerow(
                    [root.split(slashes)[-1]+slashes+name, root.split(slashes)[-1]])

    return labels


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
