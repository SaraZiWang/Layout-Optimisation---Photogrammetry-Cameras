import matplotlib.pyplot as plt
from skimage import data, img_as_float, exposure
from skimage import measure
import numpy as np
import os
import skimage as ski
from skimage.color import rgb2gray
from datetime import datetime

#You only need to define the function once and then use it anywhere you need
def polygon_area(contour):
    contour = contour.astype(float)
    x = contour[:, 1]  # x-coordinates
    y = contour[:, 0]  # y-coordinates
    # Apply shoelace formula
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def ImgBasedVisibility(rgb_img,full_img, graph=False):

    # The function needs three input
    # rgb_img is the image with object painted black, imported with ski.io.imread
    # full_img is the image with object painted black and isolated, imported with ski.io.imread
    # optional: graph is a boolean, if True, it will plot graphs, if False, it will not output any graph. If using in iterative process, turn off graph 

    start_time = datetime.now()

    # --------- IMAGE 1 - FRAME WITH OBSTRUCTION --------- #
    image = rgb2gray(rgb_img) #change image from colour to grayscale

    # Find contours at a constant value of 0.1 - chosen because from histogram, know that frame should be only area below this intensity value
    contours = measure.find_contours(image, 0.1)

    # Sort contours by size
    contours_sorted = sorted(contours, key=lambda c: len(c), reverse=True)

    # Keep only the two largest contours
    top_2_contours = contours_sorted[:2]

    # Sort contours by area in descending order
    #contour_areas_sorted = sorted(contour, key=lambda x: x[1], reverse=True)

    # Keep only the two largest areas
    top_2_areas = [polygon_area(contour) for contour in top_2_contours]

    # Calculate total area of frame
    total_area = sum(top_2_areas)
    area_rounded1 = round(total_area, 2)
    print(f"Total area of frame with obstruction is: {area_rounded1} pixels")

    # # Create figure - image with contour and histogram 
    if graph == True:
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 4))
        ax0.imshow(image,cmap=plt.cm.gray)

        # Display contours on image
        for contour in top_2_contours:
            ax0.plot(contour[:, 1], contour[:, 0], linewidth=1)
        ax0.set_axis_off()
        ax0.set_title("Contour - frame with obstruction", fontsize=12)

        # Create histrogram 
        ax1.set_title('Grayscale Histogram - with obstruction')
        ax1.set_xlabel('Grayscale Value')
        ax1.set_ylabel('Pixels')
        ax1.set_xlim([0.0, 1.0])
        ax1.hist(image.flatten(), bins=256, range=(0,1))
        plt.show()


    # --------- IMAGE 2 - FRAME WITHOUT OBSTRUCTION --------- #
    image2 = rgb2gray(full_img) #change image from colour to grayscale

    # Find contours at a constant value of 0.1 - chosen because from histogram, know that frame should be only area below this intensity value
    contours = measure.find_contours(image2, 0.1)

    # Sort contours by size
    contours_sorted = sorted(contours, key=lambda c: len(c), reverse=True)

    # Keep only the two largest contours
    top_2_contours = contours_sorted[:2]

    # Sort contours by area in descending order
    # contour_areas_sorted = sorted(contour, key=lambda x: x[1], reverse=True)

    # Keep only the two largest areas
    top_2_areas = [polygon_area(contour) for contour in top_2_contours]

    # Define the larger and smaller contour area
    largest_value = np.max(top_2_areas)
    smallest_value = np.min(top_2_areas)

    # Calculate total area of frame
    total_area = largest_value - smallest_value
    area_rounded2 = round(total_area, 2)
    print(f"Total area of frame without obstruction is: {area_rounded2} pixels")


    if graph == True:
        # Create figure - image with contour and histogram 
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 4))
        ax0.imshow(image2,cmap=plt.cm.gray)

        # Display contours on image
        for contour in top_2_contours:
            ax0.plot(contour[:, 1], contour[:, 0], linewidth=1)
        ax0.set_axis_off()
        ax0.set_title("Contour - frame without obstruction", fontsize=12)

        # Create histrogram 
        ax1.set_title('Grayscale Histogram - without obstruction')
        ax1.set_xlabel('Grayscale Value')
        ax1.set_ylabel('Pixels')
        ax1.set_xlim([0.0, 1.0])
        ax1.hist(image2.flatten(), bins=256, range=(0,1));
        plt.show()


    # --------- VISIBILITY RATIO --------- #

    # Determine the larger and smaller area
    larger_area = max(area_rounded1, area_rounded2)
    smaller_area = min(area_rounded1, area_rounded2)

    # Find frame visibility
    percentage_visibility = (smaller_area / larger_area) * 100

    frame_visibility = round(percentage_visibility, 2)
    
    print(f'Frame Visibility: {frame_visibility}%')

    end_time = datetime.now()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")

    # Outputs frame visibility and execution time (Sara)
    return frame_visibility, execution_time
