import glob
import json
import os
import time

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2

def to_Celsius(img):
    return np.round((img-27315)/100, 4)


def abs_to_Celsius_list(img_list):
    out_list = []
    for img in img_list:
        out_list.append(to_Celsius(img))
    return out_list


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)

def masking(img, lower_thd, upper_thd):
    imgnp = np.zeros((120, 160))
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > upper_thd:
                imgnp[i][j] = 1 # upper_thd / upper_thd
            elif img[i][j] > lower_thd:
                imgnp[i][j] = (img[i][j] - lower_thd) / (upper_thd-lower_thd)
            else:
                imgnp[i][j] = 0
#    return image_histogram_equalization(imgnp)
    return np.array(imgnp)

def hist_equal(input_list):
    output_list = []
    for img in input_list:
        output_list.append(image_histogram_equalization(img))
    return output_list

"""
def load_data_from_label(load_dir):
    with open(load_dir, "r") as json_file:
        label = json.load(json_file)
    pos = label["1"]
    neg = label["0"]
    return pos, neg
"""


def load_tiffs(dir_list):
    img_list = []
    for dir in dir_list:
        img_list.append(tiff.imread(dir))
    return img_list


def min_max_cutout(input_list, lower_thd, upper_thd):
    out_list = []
    for img in input_list:
        out_list.append(masking(img, lower_thd, upper_thd))
    return out_list


def amp(input_list, amplitude):
    out_list = []
    for img in input_list:
        out_list.append(np.rint(amplitude * img))
    return out_list


"""
def imshow_list(img_list):
    N = len(img_list)
    X = 2
    Y = round(N/2 + 0.1)
    cnt = 0
    for y in range(Y):
        for x in range(X):
            plt.subplot(Y, X, cnt+1)
            plt.imshow(img_list[cnt])
            plt.colorbar()
            cnt += 1
    plt.show()

def masking_imshow(array):
    lower_thd_list = [24, 24.5, 25]
    for i in range(len(array)):
        if(i % 100 == 0):
            img_list = []
            cur_image = array[i]
            for lower_thd in lower_thd_list:
                img_list.append(masking(cur_image, lower_thd, 33))
            imshow_list(img_list)

"""

def find_mean(array):
    return np.mean(array)

def find_std(array):
    stdlist = []
    for i in range(array.shape[0]):
        stdlist.append(np.std(array[i]))
    return sum(stdlist)/len(stdlist)

def standardization(img_list):
    array = np.array(img_list)
    avg = find_mean(array)
    std = find_std(array)
    print("Mean: {}, STD: {}".format(avg, std))
#    output_list = []
#    for i in range(array.shape[0]):
#        output_list.append((array[i] - avg) / std)
#    return output_list


if __name__ == "__main__":
    # Set root dirs
    root_dir = "D:/IR_data/211207_pilot_ir_data/raw_data/16bit/"
    save_dir = "D:/IR_data/211207_pilot_ir_data/raw_data/16bit_240_280/"
    load_dir = "D:/IR_data/211207_pilot_ir_data/labeled/8bit/label.json"
    lower_thd = 24
    upper_thd = 30

    dir_list = glob.glob(root_dir + '*.tiff')
#    ids = [0, 500, 1000, 6000]
#    new_list = []
#    for id in ids:
#        new_list.append(dir_list[id])
#    dir_list = new_list
    print("loaded data:{}".format(len(dir_list)))

    # 1. Data load (output range 0:65536)
    img_list = load_tiffs(dir_list)
    print("Data load done", img_list[0][0][0:10])

    # 2. Nomalization (output range ~0:100)
    img_list = abs_to_Celsius_list(img_list)
    print("Data norm done:", img_list[0][0][0:10])

    # 3. Masking (0:1)
    img_list = min_max_cutout(img_list, lower_thd, upper_thd)
    print("Masking done", img_list[0][0][0:10])

    # 4. Standartization ( -var:var )
    standardization(img_list)
#    print("Standardiation done", img_list[0][0])

    # 5. To image (0:255)
#    img_list = amp(img_list, 255)
#    print("To image done", img_list[0][0][0:10])

    # 5. Numpy to bmp
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save = True
    plot_img = False
    cnt = 0
    plt.figure(figsize=(8, 5))
    for dir in dir_list:
        array = np.repeat(np.expand_dims(img_list[cnt], axis=2), axis=2, repeats=3)
        cnt += 1

        if(save == True):
            save_path = save_dir + dir.split("\\")[-1].split(".tiff")[0] + ".bmp"
            cv2.imwrite(save_path, array*255)

        if(plot_img == True):
            plt.subplot(2, 2, cnt)
            plt.imshow(array)

    plt.show()




#        for i in range(0, len(array), 1000):
#        plt.figure(figsize=(10, 5))
#        plt.subplot(1, 2, 1)
#        plt.imshow(array)
#        plt.title("Normalized")
#        plt.colorbar()
#        plt.show()


    """
    for i in range(0, len(pos_array), 1000):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(pos_array[i])
        plt.title("Pos Raw data")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(pos_array_st[i])
        plt.colorbar()
        plt.title("Normalized")
        plt.show()

    for i in range(0, len(neg_array), 1000):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(neg_array[i])
        plt.title("Neg Raw data")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(neg_array_st[i])
        plt.colorbar()
        plt.title("Normalized")
        plt.show()

    print("done")
    """

