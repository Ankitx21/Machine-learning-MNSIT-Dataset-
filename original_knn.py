import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cv2

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define the threshold value
threshold_value = 127

# Threshold the images
x_train_thresholded = np.where(x_train > threshold_value, 1, 0)
x_test_thresholded = np.where(x_test > threshold_value, 1, 0)

### BOUNDING BOX

def construct_bounding_box(image):
    # Compute row-wise and column-wise sums of the thresholded image
    row_sums = np.sum(image, axis=1)
    col_sums = np.sum(image, axis=0)

    # Find the range of ink pixels along each row and column
    row_nonzero = np.nonzero(row_sums)[0]
    col_nonzero = np.nonzero(col_sums)[0]
    if len(row_nonzero) == 0 or len(col_nonzero) == 0:
        return np.zeros((20, 20))
    row_range = row_nonzero[[0, -1]]
    col_range = col_nonzero[[0, -1]]

    # Compute the center of the ink pixel ranges
    row_center = (row_range[0] + row_range[-1]) / 2
    col_center = (col_range[0] + col_range[-1]) / 2

    # Compute starting and ending indices for the bounding box
    row_start = int(np.clip(row_center - 9, 0, image.shape[0] - 20))
    row_end = row_start + 20
    col_start = int(np.clip(col_center - 9, 0, image.shape[1] - 20))
    col_end = col_start + 20

    # Extract the bounding box from the image
    bounding_box = image[row_start:row_end, col_start:col_end]

    return bounding_box

def construct_bounding_box_stretched(image):
    # Compute row-wise and column-wise sums of the thresholded image
    row_sums = np.sum(image, axis=1)
    col_sums = np.sum(image, axis=0)

    # Find the range of ink pixels along each row and column
    row_nonzero = np.nonzero(row_sums)[0]
    col_nonzero = np.nonzero(col_sums)[0]
    if len(row_nonzero) == 0 or len(col_nonzero) == 0:
        return np.zeros((20, 20))

    # Compute the horizontal and vertical ink pixel ranges
    row_range = row_nonzero[[0, -1]]
    col_range = col_nonzero[[0, -1]]
    row_start, row_end = row_range[0], row_range[-1]
    col_start, col_end = col_range[0], col_range[-1]

    # Stretch the extracted image to 20x20 dimensions
    image = image[row_start:row_end, col_start:col_end]
    image = resize(image, (20, 20))

    return image

x_train_bounding_box = np.zeros((len(x_train_thresholded), 20, 20))
x_train_bounding_box_stretched = np.zeros((len(x_train_thresholded), 20, 20))
for i in range(len(x_train_thresholded)):
    x_train_bounding_box[i] = construct_bounding_box(x_train_thresholded[i])
    x_train_bounding_box_stretched[i] = construct_bounding_box_stretched(x_train_thresholded[i])

x_test_bounding_box = np.zeros((len(x_test_thresholded), 20, 20))
x_test_bounding_box_stretched = np.zeros((len(x_test_thresholded), 20, 20))
for i in range(len(x_test_thresholded)):
    x_test_bounding_box[i] = construct_bounding_box(x_test_thresholded[i])
    x_test_bounding_box_stretched[i] = construct_bounding_box_stretched(x_test_thresholded[i])


from sklearn.neighbors import KNeighborsClassifier

# Reshape the datasets to fit the KNN input format
X_train_bb = x_train_bounding_box.reshape(len(x_train_bounding_box), -1)
X_train_bb_stretch = x_train_bounding_box_stretched.reshape(len(x_train_bounding_box_stretched), -1)

X_test_bb = x_test_bounding_box.reshape(len(x_test_bounding_box), -1)
X_test_bb_stretch = x_test_bounding_box_stretched.reshape(len(x_test_bounding_box_stretched), -1)



x_train_bounding_box = np.zeros((len(x_train_thresholded), 20, 20))
x_train_bounding_box_stretched = np.zeros((len(x_train_thresholded), 20, 20))
for i in range(len(x_train_thresholded)):
    x_train_bounding_box[i] = construct_bounding_box(x_train_thresholded[i])
    x_train_bounding_box_stretched[i] = construct_bounding_box_stretched(x_train_thresholded[i])

x_test_bounding_box = np.zeros((len(x_test_thresholded), 20, 20))
x_test_bounding_box_stretched = np.zeros((len(x_test_thresholded), 20, 20))
for i in range(len(x_test_thresholded)):
    x_test_bounding_box[i] = construct_bounding_box(x_test_thresholded[i])
    x_test_bounding_box_stretched[i] = construct_bounding_box_stretched(x_test_thresholded[i])

# fig, axs = plt.subplots(4, 5, figsize=(10, 5))
# axs = axs.ravel()
# for i in range(5):
#     axs[i].imshow(x_train[i], cmap='gray')
#     axs[i].set_title('Original Image')
#     axs[i+5].imshow(x_train_thresholded[i], cmap='gray')
#     axs[i+5].set_title('Thresholded Image')
#     axs[i+10].imshow(x_train_bounding_box[i], cmap='gray')
#     axs[i+10].set_title('Bounding Box Image')
#     axs[i+15].imshow(x_train_bounding_box_stretched[i], cmap='gray')
#     axs[i+15].set_title('Stretch Bounding Box Image')
# plt.show()



## Task 2

from sklearn.neighbors import KNeighborsClassifier

x_train_thresholded_flattened = x_train_thresholded.reshape(len(x_train_thresholded), -1)
x_test_thresholded_flattened = x_test_thresholded.reshape(len(x_test_thresholded), -1)

# Reshape the datasets to fit the KNN input format
X_train_bb = x_train_bounding_box.reshape(len(x_train_bounding_box), -1)
X_train_bb_stretch = x_train_bounding_box_stretched.reshape(len(x_train_bounding_box_stretched), -1)

X_test_bb = x_test_bounding_box.reshape(len(x_test_bounding_box), -1)
X_test_bb_stretch = x_test_bounding_box_stretched.reshape(len(x_test_bounding_box_stretched), -1)

# Initialize KNN classifier

# Flatten the images for input to KNN
x_train_thresholded_flattened = x_train_thresholded.reshape(len(x_train_thresholded), -1)
x_test_thresholded_flattened = x_test_thresholded.reshape(len(x_test_thresholded), -1)

k = 3
knn_bb = KNeighborsClassifier(n_neighbors=k)
knn_bb_stretch = KNeighborsClassifier(n_neighbors=k)
knn_thresholded= KNeighborsClassifier(n_neighbors=k)


# Fit the KNN models with the training data
knn_bb.fit(X_train_bb, y_train)
knn_bb_stretch.fit(X_train_bb_stretch, y_train)
knn_thresholded.fit(x_train_thresholded_flattened, y_train)

# Predict the labels of the test data using the trained models
y_pred_bb = knn_bb.predict(X_test_bb)
y_pred_bb_stretch = knn_bb_stretch.predict(X_test_bb_stretch)
y_pred_thresholded = knn_thresholded.predict(x_test_thresholded_flattened)

# Evaluate the accuracy of the models
accuracy_bb = knn_bb.score(X_test_bb, y_test)
accuracy_bb_stretch = knn_bb_stretch.score(X_test_bb_stretch, y_test)
accuracy_thresholded = knn_thresholded.score(x_test_thresholded_flattened, y_test)

# Print the accuracy scores
print(f"Accuracy of bounding box KNN: {accuracy_bb:.4f}")
print(f"Accuracy of stretched bounding box KNN: {accuracy_bb_stretch:.4f}")
print(f'Accuracy with thresholded images: {accuracy_thresholded}')

# for traning dataset:

# Flatten the images for input to KNN
x_train_thresholded_flattened = x_train_thresholded.reshape(len(x_train_thresholded), -1)

k = 3
knn_bb = KNeighborsClassifier(n_neighbors=k)
knn_bb_stretch = KNeighborsClassifier(n_neighbors=k)
knn_thresholded= KNeighborsClassifier(n_neighbors=k)

# Fit the KNN models with the training data
knn_bb.fit(X_train_bb, y_train)
knn_bb_stretch.fit(X_train_bb_stretch, y_train)
knn_thresholded.fit(x_train_thresholded_flattened, y_train)

# Predict the labels of the training data using the trained models
y_train_pred_bb = knn_bb.predict(X_train_bb)
y_train_pred_bb_stretch = knn_bb_stretch.predict(X_train_bb_stretch)
y_train_pred_thresholded = knn_thresholded.predict(x_train_thresholded_flattened)

# Evaluate the accuracy of the models on the training data
accuracy_train_bb = knn_bb.score(X_train_bb, y_train)
accuracy_train_bb_stretch = knn_bb_stretch.score(X_train_bb_stretch, y_train)
accuracy_train_thresholded = knn_thresholded.score(x_train_thresholded_flattened, y_train)

# Print the accuracy scores for the training data
print(f"Accuracy of bounding box KNN on training data: {accuracy_train_bb:.4f}")
print(f"Accuracy of stretched bounding box KNN on training data: {accuracy_train_bb_stretch:.4f}")
print(f'Accuracy with thresholded images on training data: {accuracy_train_thresholded}')

















