#%%
import cv2
import numpy as np

# Read the ground truth image
ground_truth = cv2.imread('./images/DALLE-camel-truth.png', cv2.IMREAD_GRAYSCALE)

# Convert ground truth to binary 
_, ground_truth_binary = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)

# Define a 3x3 kernel
kernel = np.ones((3, 3), np.uint8)

# Dilate 
dilated_ground_truth = cv2.dilate(ground_truth_binary, kernel, iterations=1)

# Save
cv2.imwrite('./images/DALLE-camel-truth-dilated.png', dilated_ground_truth)
#%%