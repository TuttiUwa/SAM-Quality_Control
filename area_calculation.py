# file objective: This file is to give us accurate results about the area of a polygon (that is max matrix)
# This will be used to calculate the area of void and component from the mask matrix
# function name is "area"
# the input to the function is a matrix
# the output of the function is an area (type float)
# script owner: Daniella (PGE 4)
# collaborator: Tchako Bryan (PGE 4)
# script date creation: 23/05/2023

"""
all necessary importations
"""
import cv2
import numpy as np

def area(mask):
    '''
    function to calculate the area of a polygon
    :param mask: polygon mask matrix
    :return: area: float
    '''
    # test for input
    assert isintance(mask,str)
    assert isinstance(mask, int)
    assert not isinstance(mask, np.matrix)

    # Convert the mask matrix into a binary image
    binary_image = np.zeros_like(mask, dtype=np.uint8)
    binary_image[mask > 0] = 255

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the area of the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        print("Area of the mask:", area)
    else:
        print('no countours')

    return area
