import os
import textwrap
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image as PilImage


def get_path(req_path):
    return os.path.realpath(os.path.join(os.path.dirname(__file__), req_path))


def createLabelsCsv(ds_path, csv_name):
    path_set = get_path(ds_path)
    folder_walk = os.walk(path_set)
    labels = next(folder_walk)[1]

    slashes = "/"

    if(os.name == "nt"):
        # i am on windows
        slashes = "\\"

    with open(os.path.realpath(os.path.join(path_set, csv_name)), "w") as f:
        csvwriter = csv.writer(f)
        for root, dirs, files in os.walk(path_set, topdown=True):
            for name in files:
                csvwriter.writerow(
                    [root.split(slashes)[-1]+slashes+name, root.split(slashes)[-1]])

    return labels


def get_latest_model(model_name):
    model_path = get_path(f'../../models/{model_name}')
    folder_walk = os.walk(model_path)
    return get_path(model_path + "/" + next(folder_walk)[2][-1])


def display_images(
        images: [PilImage],
        columns=4, width=20, height=8, max_images=11,
        label_wrap_length=50, label_font_size=8):

    if not images:
        print("No images to display.")
        return

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images = images[0:max_images]

    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)

        if hasattr(image, 'filename'):
            title = image.filename
            if title.endswith("/"):
                title = title[0:-1]
            title = os.path.basename(title)
            title = textwrap.wrap(title, label_wrap_length)
            title = "\n".join(title)
            plt.title(title, fontsize=label_font_size)

    plt.show()


def imshow(img):
    npimg = img.numpy()[0]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
