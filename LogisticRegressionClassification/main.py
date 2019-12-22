import numpy as np
from helper import *

# Load images/labels
t_images = get_images('tumor_images/')
t_masks = get_images('tumor_masks/')
v_images = get_images('tumor_images_val/')
v_masks = get_images('tumor_masks_val/')
t_labels = get_labels(t_masks).reshape(t_masks.shape[0],1).T
v_labels = get_labels(v_masks).reshape(v_masks.shape[0],1).T

# Flatten
training_images = t_images.reshape(t_images.shape[1] ** 2 * 3, t_images.shape[0])
validation_images = v_images.reshape(v_images.shape[1] ** 2 * 3, v_images.shape[0])

# Normalize data
training_images = training_images / 255
validation_images = validation_images / 255

# Visualize Dimensions
print("Training Dataset: {}".format(training_images.shape))
print("Validation Dataset: {}".format(validation_images.shape))
print("Training Labels: {}".format(t_labels.shape))
print("Validation Labels: {}".format(v_labels.shape))

# Logistic Regression Model
def model(X_train, Y_train, X_test, Y_test, num_iter, learning_rate):

    w, b = initialize_zeros(X_train.shape[0])
    w, b, dw, db, cost = optimize(w, b, X_train, Y_train, num_iter, learning_rate)

    y_pred_train = predict(w, b, X_train)
    y_pred_test = predict(w, b, X_test)

    train_accuracy = 100 - np.mean(np.abs(y_pred_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(y_pred_test - Y_test)) * 100

    print("Train Accuracy: {}".format(train_accuracy))
    print("Test Accuracy: {}".format(test_accuracy))

if __name__ == "__main__":

    model(training_images, t_labels, validation_images, v_labels, num_iter = 1000, learning_rate = 0.005)
