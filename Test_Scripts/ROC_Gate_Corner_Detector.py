from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
<<<<<<< HEAD
import Template_Matching_Thresholding
=======
from Template_Matching_Thresholding import template_matching_thresholding
>>>>>>> 563f0c6c19c0aecdcbc17993ea25cc30066f38b0

if sys.version_info[0] < 3:
    warnings.warn("This script should run using Python 3, which is currently not the case. The plot might not generate correctly.")

path = '../../WashingtonOBRace/'

<<<<<<< HEAD
=======

>>>>>>> 563f0c6c19c0aecdcbc17993ea25cc30066f38b0
def my_obstacle_filter(im, param):
    """ An example of a filter that can be used to generate ROC curves.
    This filter is a simple color filter around orange. Replace this with your own filter.

    :param im: image to be filtered
    :param param: filter parameter that will be varied between 0.0 and 1.0
    :return: filtered image where detected objects have a value of [255,255,255]
    """
    # Set up the filter based on the input parameter
    filter_width = np.array([0, 0, 0]) + param*255
    color_orange = np.array([255, 69, 0])

    w, h = im.size

    # Load pixel data
    im_pixels = np.asarray(im.getdata(), dtype=int)

    # Create mask for detected obstacles
    mask = np.all(im_pixels > color_orange - filter_width, axis=1) * np.all(im_pixels < color_orange + filter_width,
                                                                            axis=1)
    # Create image where obstacles are white
    filtered_im = Image.new('RGB', (w, h), color=(0, 0, 0))
    filtered_im_pixels = np.asarray(filtered_im.getdata())
    filtered_im_pixels[mask] = [255, 255, 255]
    filtered_im.putdata([tuple(pix) for pix in filtered_im_pixels])

    return filtered_im


def generate_ROC_plot():
    """ Generates a simple ROC plot"""
    plot_data = []
    n_images = 438    # Number of images in folder
<<<<<<< HEAD
    for param in np.linspace(0.5, 1.0, 11):
=======
    for param in np.linspace(0.6, 0.9, 10):
>>>>>>> 563f0c6c19c0aecdcbc17993ea25cc30066f38b0
    #for i in range(1):
        # Initialize totals
        true_positives = 0
        false_positives = 0
        ground_truth_positives = 0
        ground_truth_negatives = 0
        
        print('Current parameter: ', param)
        template_matching_thresholding(param)

        for i in range(1, n_images + 1):
            # Set image paths
            #original_path = path + 'WashingtonOBRace/img_' + str(i) + '.png'
<<<<<<< HEAD
            ground_truth_path = path + 'WashingtonOBRace/Scaled_Masks/mask_' + str(i) + '.png'
=======
            ground_truth_path = path + 'WashingtonOBRace/mask_' + str(i) + '.png'
>>>>>>> 563f0c6c19c0aecdcbc17993ea25cc30066f38b0
            filter_path = path + 'Output/mask_' + str(i) + '.png'

            # Analyze ground truth image
            try:
                ground_truth_im = Image.open(ground_truth_path, 'r')
            except:
<<<<<<< HEAD
                #print('image ', i, ' exception')
=======
                print('image ', i, ' exception')
>>>>>>> 563f0c6c19c0aecdcbc17993ea25cc30066f38b0
                continue
            
            ground_truth_im = ground_truth_im.convert("RGB")
            ground_truth_pixels = np.asarray(ground_truth_im.getdata())
            ground_truth_obstacles = np.all(ground_truth_pixels == [255, 255, 255], axis=1)

            # Analyze original image
            #im = Image.open(original_path, 'r')
            #filtered_im = my_obstacle_filter(im, param)
            filtered_im = Image.open(filter_path, 'r')
            filtered_im_pixels = np.asarray(filtered_im.getdata())
            filtered_im_obstacles = np.all(filtered_im_pixels == [255, 255, 255], axis=1)

            # Update totals of positives/negatives
            true_positives += np.sum((filtered_im_obstacles == True) & (ground_truth_obstacles == True))
            false_positives += np.sum((filtered_im_obstacles == True) & (ground_truth_obstacles == False))

            ground_truth_positives += np.sum((ground_truth_obstacles == True))
            ground_truth_negatives += np.sum((ground_truth_obstacles == False))
            
            #print(int(round(i/n_images*100, 0)), ' %')

        # Calculate rates
        false_positive_rate = false_positives / ground_truth_negatives
        true_positive_rate = true_positives / ground_truth_positives
        print('False Positive: ', false_positive_rate)
        print('True Positive: ', true_positive_rate)

        # Add datapoint to plot_data
        plot_data.append((false_positive_rate, true_positive_rate))

    # Create x and y data from plot_data
    x = [item[0] for item in plot_data]
    y = [item[1] for item in plot_data]

    # Plot
    plt.plot(x, y)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()


# Main script
generate_ROC_plot()
