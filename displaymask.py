# all necessary importation for the script to run properly
import numpy as np


# method to visualize  generated mask
def show_mask(mask, ax, random_color=False):
    """
    function to display mask
    :param mask:
    :param ax:
    :param random_color:
    :return: none
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


