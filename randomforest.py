from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from keras.datasets import mnist
from skimage.transform import resize

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


# Flatten the bounding box images
x_train_bounding_box_flat = x_train_bounding_box.reshape(len(x_train_bounding_box), -1)
x_train_bounding_box_stretched_flat = x_train_bounding_box_stretched.reshape(len(x_train_bounding_box_stretched), -1)
x_test_bounding_box_flat = x_test_bounding_box.reshape(len(x_test_bounding_box), -1)
x_test_bounding_box_stretched_flat = x_test_bounding_box_stretched.reshape(len(x_test_bounding_box_stretched), -1)
x_train_thresholded_flat = x_train_thresholded.reshape(len(x_train_thresholded), -1)
x_test_thresholded_flat = x_test_thresholded.reshape(len(x_test_thresholded), -1)


from sklearn.ensemble import RandomForestClassifier

# # for trees = 10 and depth = 4

# # Define the classifier
# rfc = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)


# # Fit the classifier on the training data
# rfc.fit(x_train_thresholded_flat, y_train)

# # Make predictions on the test data
# y_pred_thresholded = rfc.predict(x_train_thresholded_flat)
# accuracy_thresholded = accuracy_score(y_train, y_pred_thresholded)


# # Fit the classifier on the training data
# rfc.fit(x_train_bounding_box_flat, y_train)

# # Make predictions on the test data
# y_pred_bounding_box = rfc.predict(x_train_bounding_box_flat)
# accuracy_bounding_box = accuracy_score(y_train, y_pred_bounding_box)


# # Fit the classifier on the training data
# rfc.fit(x_train_bounding_box_stretched_flat, y_train)

# # Make predictions on the test data
# y_pred_bounding_box_stretched = rfc.predict(x_train_bounding_box_stretched_flat)
# accuracy_bounding_box_stretched = accuracy_score(y_train, y_pred_bounding_box_stretched)


# print(" Accuracy for trees 🌲 = 10 and depth =4")
# print("Accuracy of thresholded images:", accuracy_thresholded)
# print("Accuracy of bounding box images:", accuracy_bounding_box)
# print("Accuracy of stretched bounding box images:", accuracy_bounding_box_stretched)

# # # for trees = 10 and depth = 16

# # Define the classifier
# rfc = RandomForestClassifier(n_estimators=10, max_depth=16, random_state=42)


# # Fit the classifier on the training data
# rfc.fit(x_train_thresholded_flat, y_train)

# # Make predictions on the test data
# y_pred_thresholded = rfc.predict(x_train_thresholded_flat)
# accuracy_thresholded = accuracy_score(y_train, y_pred_thresholded)


# # Fit the classifier on the training data
# rfc.fit(x_train_bounding_box_flat, y_train)

# # Make predictions on the test data
# y_pred_bounding_box = rfc.predict(x_train_bounding_box_flat)
# accuracy_bounding_box = accuracy_score(y_train, y_pred_bounding_box)


# # Fit the classifier on the training data
# rfc.fit(x_train_bounding_box_stretched_flat, y_train)

# # Make predictions on the test data
# y_pred_bounding_box_stretched = rfc.predict(x_train_bounding_box_stretched_flat)
# accuracy_bounding_box_stretched = accuracy_score(y_train, y_pred_bounding_box_stretched)


# print(" Accuracy for trees 🌲 = 10 and depth =16")
# print("Accuracy of thresholded images:", accuracy_thresholded)
# print("Accuracy of bounding box images:", accuracy_bounding_box)
# print("Accuracy of stretched bounding box images:", accuracy_bounding_box_stretched)

# # # for trees = 30 and depth = 4

# # Define the classifier
# rfc = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)


# # Fit the classifier on the training data
# rfc.fit(x_train_thresholded_flat, y_train)

# # Make predictions on the test data
# y_pred_thresholded = rfc.predict(x_train_thresholded_flat)
# accuracy_thresholded = accuracy_score(y_train, y_pred_thresholded)


# # Fit the classifier on the training data
# rfc.fit(x_train_bounding_box_flat, y_train)

# # Make predictions on the test data
# y_pred_bounding_box = rfc.predict(x_train_bounding_box_flat)
# accuracy_bounding_box = accuracy_score(y_train, y_pred_bounding_box)


# # Fit the classifier on the training data
# rfc.fit(x_train_bounding_box_stretched_flat, y_train)

# # Make predictions on the test data
# y_pred_bounding_box_stretched = rfc.predict(x_train_bounding_box_stretched_flat)
# accuracy_bounding_box_stretched = accuracy_score(y_train, y_pred_bounding_box_stretched)


# print(" Accuracy for trees 🌲 = 30 and depth =4")
# print("Accuracy of thresholded images:", accuracy_thresholded)
# print("Accuracy of bounding box images:", accuracy_bounding_box)
# print("Accuracy of stretched bounding box images:", accuracy_bounding_box_stretched)


# Define the classifier
rfc = RandomForestClassifier(n_estimators=30, max_depth=16, random_state=42)


# Fit the classifier on the training data
rfc.fit(x_train_thresholded_flat, y_train)

# Make predictions on the test data
y_pred_thresholded = rfc.predict(x_train_thresholded_flat)
accuracy_thresholded = accuracy_score(y_train, y_pred_thresholded)


# Fit the classifier on the training data
rfc.fit(x_train_bounding_box_flat, y_train)

# Make predictions on the test data
y_pred_bounding_box = rfc.predict(x_train_bounding_box_flat)
accuracy_bounding_box = accuracy_score(y_train, y_pred_bounding_box)


# Fit the classifier on the training data
rfc.fit(x_train_bounding_box_stretched_flat, y_train)

# Make predictions on the test data
y_pred_bounding_box_stretched = rfc.predict(x_train_bounding_box_stretched_flat)
accuracy_bounding_box_stretched = accuracy_score(y_train, y_pred_bounding_box_stretched)


print(" Accuracy for trees 🌲 = 30 and depth =16")
print("Accuracy of thresholded images:", accuracy_thresholded)
print("Accuracy of bounding box images:", accuracy_bounding_box)
print("Accuracy of stretched bounding box images:", accuracy_bounding_box_stretched)