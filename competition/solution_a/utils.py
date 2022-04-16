from os import path, walk
import csv


def get_path(req_path):
    return path.realpath(path.join(path.dirname(__file__), req_path))


def createLabelsCsv(ds_path, csv_name):
    path_set = get_path(ds_path)
    folder_walk = walk(path_set)
    labels = next(folder_walk)[1]
    with open(path.join(path.dirname(__file__), csv_name), "w") as f:
        csvwriter = csv.writer(f)
        for root, dirs, files in walk(path_set, topdown=True):
            for name in files:
                csvwriter.writerow(
                    [root.split("\\")[-1]+"\\"+name, root.split("\\")[-1]])
