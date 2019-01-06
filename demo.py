import model as m
import voters as v

import argparse
import matplotlib.pyplot as plt 

import skimage
from skimage import io as skio

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="path to input image file", default="image.jpg")
    parser.add_argument("-d", "--dict", help="path to saved dictionary", default="model.pth")
    args = vars(parser.parse_args())

    image = skio.imread(args["image"])

    corrected_images = []
    # GW
    corrected_images.append(v.correct_image(image, v.grey_edge(image, njet=0, mink_norm=1, sigma=0)))
    # MAX-RGB
    corrected_images.append(v.correct_image(image, v.grey_edge(image, njet=0, mink_norm=-1, sigma=0)))
    # GE
    corrected_images.append(v.correct_image(image, v.grey_edge(image, njet=1, mink_norm=5, sigma=2)))
    # CNN
    model = m.ColorConstancyModel([(9, 27), (27, 3)])
    model.load_state_dict(torch.load(args["dict"]))
    corrected_images.append(v.correct_image(image, m.predict(model, image)))

    # show results
    rows = 3
    cols = 2
    fig = plt.figure()

    fig.add_subplot(rows, cols, 1)
    plt.imshow(image)
    plt.title('Original Image')

    fig.add_subplot(rows, cols, 3)
    plt.imshow(corrected_images[0])
    plt.title('GW')

    fig.add_subplot(rows, cols, 4)
    plt.imshow(corrected_images[1])
    plt.title('MAX-RGB')

    fig.add_subplot(rows, cols, 5)
    plt.imshow(corrected_images[2])
    plt.title('GE')

    fig.add_subplot(rows, cols, 6)
    plt.imshow(corrected_images[3])
    plt.title('CNN')

    plt.show()

