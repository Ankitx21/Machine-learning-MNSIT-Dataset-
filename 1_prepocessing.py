# import numpy as np
# import matplotlib.pyplot as plt
# from keras.datasets import mnist
# import cv2

# # Define the construct_bounding_box function
# def construct_bounding_box(image):
#     # Compute row-wise and column-wise sums of the thresholded image
#     row_sums = np.sum(image, axis=1)
#     col_sums = np.sum(image, axis=0)

#     # Find the range of ink pixels along each row and column
#     row_range = np.where(row_sums > 0)[0][[0, -1]]
#     col_range = np.where(col_sums > 0)[0][[0, -1]]

#     # Compute the center of the ink pixel ranges
#     row_center = np.mean(row_range)
#     col_center = np.mean(col_range)

#     # Compute starting and ending indices for the bounding box
#     row_start = int(np.clip(row_center - 9, 0, image.shape[0] - 20))
#     row_end = row_start + 20
#     col_start = int(np.clip(col_center - 9, 0, image.shape[1] - 20))
#     col_end = col_start + 20

#     # Extract the bounding box from the image
#     bounding_box = image[row_start:row_end, col_start:col_end]

#     return bounding_box

# # Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Define the threshold value
# threshold_value = 127

# # Threshold the images
# x_train_thresholded = np.where(x_train > threshold_value, 1, 0)
# x_test_thresholded = np.where(x_test > threshold_value, 1, 0)

# # Plot some example images
# # fig, axs = plt.subplots(2, 5, figsize=(10, 5))
# # axs = axs.ravel()
# # for i in range(5):
# #     axs[i].imshow(x_train[i], cmap='gray')
# #     axs[i].set_title('Original Image')
# #     axs[i+5].imshow(x_train_thresholded[i], cmap='gray')
# #     axs[i+5].set_title('Thresholded Image')
# # plt.show()

# # Construct the bounding box for the first image in the training set
# for i in range(len(x_train_thresholded)):
#     train_bounding_box = construct_bounding_box(x_train_thresholded[i])
#     # Display the original image, thresholded image, and bounding box image
#     # fig, axs = plt.subplots(1, 3, figsize=(7, 3))
#     # axs[0].imshow(x_train[i], cmap='gray')
#     # axs[0].set_title('Original Image')
#     # axs[1].imshow(x_train_thresholded[i], cmap='gray')
#     # axs[1].set_title('Thresholded Image')
#     # axs[2].imshow(train_bounding_box, cmap='gray')
#     # axs[2].set_title('Bounding Box Image')
#     # plt.show()

# for i in range(len(x_test_thresholded)):
#     test_bounding_box = construct_bounding_box(x_test_thresholded[i])


#     # fig, axs = plt.subplots(1, 3, figsize=(7, 3))
#     # axs[0].imshow(x_test[i], cmap='gray')
#     # axs[0].set_title('Original Image')
#     # axs[1].imshow(x_test_thresholded[i], cmap='gray')
#     # axs[1].set_title('Thresholded Image')
#     # axs[2].imshow(test_bounding_box, cmap='gray')
#     # axs[2].set_title('Bounding Box Image')
#     # plt.show()


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from PIL import Image

def image_processor(image):
    threshold = 170

    # Threshold the data
    image = np.where(image < threshold, 0, 1)

    nonzero_indices = np.nonzero(image)
    min_x, max_x = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    min_y, max_y = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])

    boxed_image = image[min_y:max_y, min_x:max_x]

    # Convert the numpy array to a PIL Image object
    boxed_image = Image.fromarray(boxed_image)

    # Resize the image to 20x20
    stretched_bounding_box = boxed_image.resize((20, 20))

    # Convert the PIL Image object back to a numpy array
    stretched_bounding_box = np.array(stretched_bounding_box)

    return stretched_bounding_box

def process_images(images):
    """Process a batch of images by thresholding, cropping, padding, and resizing them."""
    processed_images = []
    for image in images:
        processed = image_processor(image)
        processed_images.append(processed)
    return np.array(processed_images)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Process the training images
processed_x_train = process_images(x_train)

# Show an example processed image
fig = plt.figure()
plt.imshow(processed_x_train[7], cmap='gray')
plt.title('Processed Image')
plt.show()
