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

""" 
folders = ["./datasets/animal_scraped/training/", "./datasets/animal_scraped/validation/gallery/", "./datasets/animal_scraped/validation/query/"]

for main_dir in folders:
    folder_walk = os.walk(main_dir)
    fs = next(folder_walk)[1]
    for folder in fs:
        filenames = next(os.walk(main_dir + folder))[2]
        for filename in filenames:
            os.rename(main_dir + folder + "/" + filename, main_dir + folder + "/" + filename.split("_")[1]) """