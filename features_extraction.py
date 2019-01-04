import utils
import voters
from voters import grey_edge

import argparse
import numpy as np

import skimage
from skimage import io as skio

import scipy
from scipy import io as scio


def read_files(base_dir, list_file_path, illums_file_path):
    # TODO: add doc here

    # get images path
    with open(list_file_path) as list_file:
        list_file_content = list_file.readlines()
    images_path = [base_dir + x.strip() for x in list_file_content]

    illums = scipy.io.loadmat(illums_file_path)['real_rgb']

    return images_path, illums


def get_votes(images_path, correct_illums, debug=False):
    # TODO: add doc here

    # constants
    GREY_WORLD = 'grey_world'
    MAX_RGB = 'max_rgb'
    GREY_EDGE = 'grey_edge'

    # keys lists
    keys = [GREY_WORLD, MAX_RGB, GREY_EDGE]
    # voters lists
    voters = {
        GREY_WORLD: lambda x: grey_edge(x, njet=0, mink_norm=1, sigma=0),
        MAX_RGB: lambda x: grey_edge(x, njet=0, mink_norm=-1, sigma=0),
        GREY_EDGE: lambda x: grey_edge(x, njet=1, mink_norm=5, sigma=2)
    }
    # illum lists
    illums = {
        GREY_WORLD: [],
        MAX_RGB: [],
        GREY_EDGE: []
    }
    # error lists
    errors = {
        GREY_WORLD: [],
        MAX_RGB: [],
        GREY_EDGE: []
    }
    # estimate illuminations and calculate error
    for index, (image_path, correct_illum) in enumerate(zip(images_path, correct_illums)):
        image = skio.imread(image_path)
        if debug:
            print(image_path)
            print('illumination: ' + str(correct_illum))

        for key in keys:
            estim_illum = voters[key](image)
            illums[key].append(estim_illum)
            errors[key].append(utils.angular_error(estim_illum, correct_illum))
            if debug:
                print(key + ": " + str(estim_illum) + " error: " + str(errors[key][-1]))
        if debug:
            print()
        else:
            utils.clear_screen()
            print(str(index + 1) + ' / ' + str(len(images_path)))

    for key in keys:
        print(key + ":")
        utils.print_stats(errors[key])

    return illums, errors


def save_features(base_dir, list_file_path, illums_file_path, debug=False):
    # TODO: add doc here
    illums, __ = get_votes(*read_files(base_dir, list_file_path, illums_file_path), debug)
    np.save('features.npy', illums) 


if __name__ == "__main__":
    # for manual testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--dir", required=True, help="path to base of the dataset")
    parser.add_argument("-l", "--list", required=True, help="path to the list file")
    parser.add_argument("-i", "--illums", required=True, help="path to correct illuminations file")
    parser.add_argument('-d', "--debug", action='store_true', help='activate debug mode', default=False)
    args = vars(parser.parse_args())

    # extract and save features to a file
    save_features(args["dir"], args["list"], args["illums"], args["debug"])
