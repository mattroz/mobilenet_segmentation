import matplotlib.pyplot as plt


# Function was forked from https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb
def visualize(image, mask, original_image=None, original_mask=None, name=None):
    """
    Function for two image-mask pairs visualizing.

    image: numpy array
        First image to visualize.
    mask: numpy array
        Mask for the first image.
    original_image: numpy array
        Second image to visualize
    original_mask: numpy array
        Mask for the second image
    name: str
        If specified, produced plot will be saved under this name
    """

    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Original image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Predicted mask', fontsize=fontsize)

    if name is not None:
        plt.savefig(name)
