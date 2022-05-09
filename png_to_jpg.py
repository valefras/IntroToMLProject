import os
from PIL import Image
import magic
mime = magic.Magic(mime=True)

folders = ["./datasets/animal_scraped/training/", "./datasets/animal_scraped/validation/gallery/", "./datasets/animal_scraped/validation/query/"]

for main_dir in folders:
    folder_walk = os.walk(main_dir)
    fs = next(folder_walk)[1]

    for folder in fs:
        filenames = next(os.walk(main_dir + folder))[2]
        for filename in filenames:
            im_path = main_dir + folder + "/" + filename
            if mime.from_file(im_path) == "image/png":
                im = Image.open(im_path)
                rgb_im = im.convert('RGB')
                rgb_im.save("." + im_path.split(".")[1] + ".jpg", quality=100)
                os.remove(os.path.realpath(im_path))
        print(f"{folder} in {main_dir} done")