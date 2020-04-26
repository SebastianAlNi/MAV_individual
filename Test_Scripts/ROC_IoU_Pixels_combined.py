from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
import Template_Matching_Thresholding as TMT
#from Template_Matching_Thresholding import template_matching_thresholding

if sys.version_info[0] < 3:
    warnings.warn("This script should run using Python 3, which is currently not the case. The plot might not generate correctly.")

path = '../../WashingtonOBRace/'

def intersection_over_union(img_gt, img_filter):
    
    # Analyze the ground truth image
    img_gt = img_gt.convert("RGB")
    ground_truth_pixels = np.asarray(img_gt.getdata())
    ground_truth_obstacles = np.all(ground_truth_pixels == [255, 255, 255], axis=1)
    img_gt_area = np.sum(ground_truth_obstacles == True)

    # Analyze filtered image
    img_filter = img_filter.convert("RGB")
    filtered_im_pixels = np.asarray(img_filter.getdata())
    filtered_im_obstacles = np.all(filtered_im_pixels == [255, 255, 255], axis=1)
    img_filter_area = np.sum(filtered_im_obstacles == True)
    
    interArea = np.sum((ground_truth_obstacles == True) & (filtered_im_obstacles == True))

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(img_gt_area + img_filter_area - interArea)

	# return the intersection over union value
    return iou, ground_truth_obstacles, filtered_im_obstacles, interArea, img_gt_area, img_filter_area


def generate_ROC_plot():
    """ Generates a simple ROC plot"""
    plot_data = []
    n_images = 438    # Number of images in folder
    iou_threshold = 0.6
    
    file = open('ROC_Output/output.txt', 'w')
    file.write('false_positive_rate\ttrue_positive_rate\n')
    file.close()
            
    case1 = 0
    case2 = 0
    case3 = 0
    
    for param in np.linspace(0.89, 1.0, 12):
        file = open('ROC_Output/output.txt', 'a')
    #for i in range(1):
        # Initialize totals
        true_positives = 0
        false_positives = 0
        ground_truth_positives = 0
        ground_truth_negatives = 0
        total = 0
        
        print('Current parameter: ', param)
        TMT.template_matching_thresholding(param)

        for i in range(1, n_images + 1):
            
            # Set image paths            
            ground_truth_path = path + 'WashingtonOBRace/Scaled_Masks/mask_' + str(i) + '.png'
            filter_path = path + 'Output/mask_' + str(i) + '.png'

            # Analyze ground truth image
            try:
                ground_truth_im = Image.open(ground_truth_path, 'r')
                filtered_im = Image.open(filter_path, 'r')
            except:
                continue
            
            iou, ground_truth_obstacles, filtered_im_obstacles, int_area, gt_area, ft_area = intersection_over_union(ground_truth_im, filtered_im)
            
            if iou >= iou_threshold:
                true_positives += 1
                case1+=1
            elif abs(int_area - gt_area) < 2:
                #true_positives += 1
                false_positives += 1
                case2+=1
            elif iou < iou_threshold and ft_area > 10:
                false_positives += 1
                case3+=1
                
            ground_truth_positives += 1 # Every ground truth frame is a positive since every frame shows a gate
            total += 1

            # Update totals of positives/negatives
            #true_positives += np.sum((filtered_im_obstacles == True) & (ground_truth_obstacles == True))
            #false_positives += np.sum((filtered_im_obstacles == True) & (ground_truth_obstacles == False))

            #ground_truth_positives += np.sum((ground_truth_obstacles == True))
            #ground_truth_negatives += np.sum((ground_truth_obstacles == False))
            
            #print(int(round(i/n_images*100, 0)), ' %')

        # Calculate rates
        #false_positive_rate = false_positives / ground_truth_negatives
        #true_positive_rate = true_positives / ground_truth_positives
        #false_positive_rate = false_positives #/ ground_truth_negatives
        false_positive_rate = false_positives / total
        true_positive_rate = true_positives / ground_truth_positives
        print('False Positive: ', false_positive_rate)
        print('True Positive: ', true_positive_rate)

        # Add datapoint to plot_data
        plot_data.append((false_positive_rate, true_positive_rate))
        
        file.write(str(false_positive_rate) + '\t' + str(true_positive_rate) + '\n')
        
    file.close()

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
    
    print(case1)
    print(case2)
    print(case3)


# Main script
generate_ROC_plot()
