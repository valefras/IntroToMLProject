import os
import magic
mime = magic.Magic(mime=True)

folder_walk = os.walk("./datasets/animal_scraped/training/")
fs = next(folder_walk)[1]

for folder in fs:
    filenames = next(os.walk("./datasets/animal_scraped/training/" + folder))[2]
    for filename in filenames:
        if mime.from_file("./datasets/animal_scraped/training/" + folder + "/" + filename) == "image/jpeg":
            os.rename("./datasets/animal_scraped/training/" + folder + "/" + filename, "./datasets/animal_scraped/training/" + folder + "/" + filename.split(".")[0] + ".jpg")
        else:
            os.rename("./datasets/animal_scraped/training/" + folder + "/" + filename, "./datasets/animal_scraped/training/" + folder + "/" + filename.split(".")[0] + ".png")