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
    
    interArea = np.sum(np.logical_and(ground_truth_obstacles == True, filtered_im_obstacles == True))

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(img_gt_area + img_filter_area - interArea)

	# return the intersection over union value
    return iou, np.sum(filtered_im_obstacles)


def generate_ROC_plot():
    """ Generates a simple ROC plot"""
    plot_data = []
    n_images = 438    # Number of images in folder
    iou = np.zeros(n_images)
    iou.fill(-1)
    fio = np.zeros(n_images)
    #iou_threshold = 0.5
    
    file = open('ROC_Output/output.txt', 'w')
    file.write('IoU Threshold\tTrue Positive Rate\tFalse Positive Rate\n')
    file.close()
    
    for param in np.linspace(0.96, 0.96, 1):
        file = open('ROC_Output/output.txt', 'a')
        
        print('Current parameter: ', param)
        #TMT.template_matching_thresholding(param)

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
            
            iou[i-1], fio[i-1] = intersection_over_union(ground_truth_im, filtered_im)
        
        #print(iou)
            
        for iou_threshold in np.linspace(0.0, 1.0, 101):
            # Initialize totals
            true_positives = 0
            false_positives = 0
            #ground_truth_positives = 0
            #ground_truth_negatives = 0
            total = 0
            
            for j in range(1, n_images + 1):
                if iou[j-1] == -1:
                    continue
                elif iou[j-1] >= iou_threshold:
                    true_positives += 1
                elif iou[j-1] < iou_threshold and fio[j-1] > 10:
                    false_positives += 1
                    
                total += 1 # Every ground truth frame is a positive since every frame shows a gate
    
                # Update totals of positives/negatives
                #true_positives += np.sum((filtered_im_obstacles == True) & (ground_truth_obstacles == True))
                #false_positives += np.sum((filtered_im_obstacles == True) & (ground_truth_obstacles == False))
    
                #ground_truth_positives += np.sum((ground_truth_obstacles == True))
                #ground_truth_negatives += np.sum((ground_truth_obstacles == False))
                
                #print(int(round(i/n_images*100, 0)), ' %')
        
            # Calculate rates
            #false_positive_rate = false_positives / ground_truth_negatives
            #true_positive_rate = true_positives / ground_truth_positives
            false_positive_rate = false_positives / total
            true_positive_rate = true_positives / total
            #print('False Positive: ', false_positive_rate)
            #print('True Positive: ', true_positive_rate)
            #print(true_positive_rate)
            # Add datapoint to plot_data
            plot_data.append((iou_threshold, true_positive_rate, false_positive_rate))
            
            file.write(str(iou_threshold) + '\t' + str(true_positive_rate) + '\t' + str(false_positive_rate) + '\n')
        
    file.close()

    # Create x and y data from plot_data
    x = [item[0] for item in plot_data]
    y = [item[1] for item in plot_data]
    y2 = [item[2] for item in plot_data]

    # Plot
    plt.plot(x, y, x, y2)
    plt.legend(['True Positive Rate', 'False Positive Rate'])
    plt.xlabel("IoU Threshold")
    plt.ylabel("True/False Positive Ratio")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()


# Main script
generate_ROC_plot()
