import argparse
import numpy as np
import matplotlib.pyplot as plt 

import skimage
from skimage import filters
from skimage import io


def grey_edge(image, njet=0, mink_norm=1, sigma=1):
    """
    Estimates the light source of an input_image as proposed in:
    J. van de Weijer, Th. Gevers, A. Gijsenij
    "Edge-Based Color Constancy"
    IEEE Trans. Image Processing, accepted 2007.
    Depending on the parameters the estimation is equal to Grey-World, Max-RGB, general Grey-World,
    Shades-of-Grey or Grey-Edge algorithm.

    :param image: rgb input image (NxMx3)
    :param njet: the order of differentiation (range from 0-2)
    :param mink_norm: minkowski norm used (if mink_norm==-1 then the max
           operation is applied which is equal to minkowski_norm=infinity).
    :param sigma: sigma used for gaussian pre-processing of input image

    :return: illuminant color estimation

    :raise: ValueError
    """

    # pre-process image by applying gauss filter
    gauss_image = filters.gaussian(image, sigma=sigma, multichannel=True)

    # get njet-order derivative of the pre-processed image
    if njet == 0:
        deriv_image = [gauss_image[:, :, channel] for channel in range(3)]
    else:   
        if njet == 1:
            deriv_filter = filters.sobel
        elif njet == 2:
            deriv_filter = filters.laplace
        else:
            raise ValueError("njet should be in range[0-2]! Given value is: " + str(njet))     
        deriv_image = [np.abs(deriv_filter(gauss_image[:, :, channel])) for channel in range(3)]

    # remove saturated pixels in input image
    for channel in range(3):
        deriv_image[channel][image[:, :, channel] >= 255] = 0.

    # estimate illuminations
    if mink_norm == -1:  # mink_norm = inf
        estimating_func = np.max 
    else:
        estimating_func = lambda x: np.power(np.sum(np.power(x, mink_norm)), 1 / mink_norm)
    illum = [estimating_func(channel) for channel in deriv_image]

    # normalize estimated illumination
    som = np.sqrt(np.sum(np.power(illum, 2)))
    illum = np.divide(illum, som)

    return illum


def correct_image(image, illum):
    """
    Corrects image colors by performing diagonal transformation according to 
    given estimated illumination of the image.
    
    :param image: rgb input image (NxMx3)
    :param illum: estimated illumination of the image

    :return: corrected image
    """

    correcting_illum = illum * np.sqrt(3)
    corrected_image = image / 255.

    for channel in range(3):
        corrected_image[:, :, channel] /= correcting_illum[channel]
    return corrected_image


if __name__ == "__main__":
    # for manual testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image file")
    args = vars(parser.parse_args())

    # load the image from disk
    input_image = io.imread(args["image"])

    # test voters
    # estimate illuminations
    grey_edge_illum = grey_edge(input_image)
    print('Grey-Edge illumination: ' + str(grey_edge_illum))
    # correct images
    corrected_image = correct_image(input_image, grey_edge_illum)

    # show results
    rows = 1
    cols = 2
    fig=plt.figure(figsize=(100, 75))

    fig.add_subplot(rows, cols, 1)
    plt.imshow(input_image)
    plt.title('Original Image')

    fig.add_subplot(rows, cols, 2)
    plt.imshow(corrected_image)
    plt.title('Corrected Image')

    plt.show()
