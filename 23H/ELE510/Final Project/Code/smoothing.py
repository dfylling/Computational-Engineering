#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the image
image = cv2.imread('./images/DALLE-camel.png', cv2.IMREAD_GRAYSCALE)

# Apply degrees of Gaussian smoothing
blurred_image_3 = cv2.GaussianBlur(image, (3, 3), 0)
blurred_image_5 = cv2.GaussianBlur(image, (5, 5), 0)
blurred_image_7 = cv2.GaussianBlur(image, (7, 7), 0)

# Edge detection
t1 = 30
t2 = 100
I_edges = cv2.Canny(image, threshold1=t1, threshold2=t2)
Ib3_edges = cv2.Canny(blurred_image_3, threshold1=t1, threshold2=t2)
Ib5_edges = cv2.Canny(blurred_image_5, threshold1=t1, threshold2=t2)
Ib7_edges = cv2.Canny(blurred_image_7, threshold1=t1, threshold2=t2)

# Load ground_truth_image
ground_truth_image = cv2.imread('./images/DALLE-camel-truth.png', cv2.IMREAD_GRAYSCALE)
ground_truth_binary = (ground_truth_image > 0).astype(np.uint8) * 255

# Create a DataFrame to store the evaluation metrics
data = {
    "Image": ["Original", "Kernel size 3x3", "Kernel size 5x5", "Kernel size 7x7"],
    "Precision": [],
    "Recall": [],
    "F1-score": []}


# Calculate evaluation metrics for each result and add to the DataFrame
average = 'weighted'
for edges, label in zip([I_edges, Ib3_edges, Ib5_edges, Ib7_edges], data["Image"]):
    edges_binary = (edges > 0).astype(np.uint8) * 255
    precision = precision_score(ground_truth_binary, edges_binary, average=average)
    recall = recall_score(ground_truth_binary, edges_binary, average=average)
    f1 = f1_score(ground_truth_binary, edges_binary, average=average)

    data["Precision"].append(precision)
    data["Recall"].append(recall)
    data["F1-score"].append(f1)

# Create a pandas DataFrame
df = pd.DataFrame(data)

print(df)

# Create a figure with subplots to display the intermediate results
plt.figure(figsize=(30, 30))
fontsize = 60

plt.subplot(221)
plt.imshow(I_edges, cmap='gray')
plt.title('Original image', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(222)
plt.imshow(Ib3_edges, cmap='gray')
plt.title('Kernel size 3x3', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(223)
plt.imshow(Ib5_edges, cmap='gray')
plt.title('Kernel size 5x5', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(224)
plt.imshow(Ib7_edges, cmap='gray')
plt.title('Kernel size 7x7', fontsize=fontsize)
plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.show()
#%%