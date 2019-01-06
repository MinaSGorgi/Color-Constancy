import model as m
import voters as v

import argparse
import matplotlib.pyplot as plt 
import os

import skimage
from skimage import io as skio

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", nargs="+", help="paths to input images",
                        default=[os.path.join("assets", image) for image in ["tungsten.png", "bluesky.png"]])
    parser.add_argument("-d", "--dict", help="path to saved dictionary", default=os.path.join("res", "model.pth"))
    args = vars(parser.parse_args())

    model = m.ColorConstancyModel([(9, 27), (27, 3)])
    model.load_state_dict(torch.load(args["dict"]))

    images = []
    for image_path in args["images"]:
        image = skio.imread(image_path)
        # Original
        images.append(image)
        # GW
        images.append(v.correct_image(image, v.grey_edge(image, njet=0, mink_norm=1, sigma=0)))
        # MAX-RGB
        images.append(v.correct_image(image, v.grey_edge(image, njet=0, mink_norm=-1, sigma=0)))
        # GE
        images.append(v.correct_image(image, v.grey_edge(image, njet=1, mink_norm=5, sigma=2)))
        # CNN
        images.append(v.correct_image(image, m.predict(model, image)))

    # show results
    rows = len(args["images"])
    cols = 5
    fig = plt.figure(figsize=(80, 80))
    titles = ["Original", "GW", "MAX-RGB", "GE", "CNN"]
    for row in range(rows):
        for col in range(cols):
            index = col + row * 5
            fig.add_subplot(rows, cols, index + 1)
            plt.imshow(images[index])
            plt.title(titles[col])
    plt.show()

