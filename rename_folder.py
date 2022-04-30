import os

from competition.utils import utils

folder_walk = os.walk("./datasets/new_animals/gallery")
fs = next(folder_walk)[1]
cont = 0
for folder in fs:
    os.rename("./datasets/new_animals/gallery/" + folder, "./datasets/new_animals/gallery/"+str(cont))
    cont += 1