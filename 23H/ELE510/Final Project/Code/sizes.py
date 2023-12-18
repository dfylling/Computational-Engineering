#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the images
image_full = cv2.imread('./images/DALLE-camel.png', cv2.IMREAD_GRAYSCALE)
image_half = cv2.imread('./images/DALLE-camel-half.png', cv2.IMREAD_GRAYSCALE)
image_quart = cv2.imread('./images/DALLE-camel-quart.png', cv2.IMREAD_GRAYSCALE)
image_eighth = cv2.imread('./images/DALLE-camel-eighth.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian smoothing
blurred_full = cv2.GaussianBlur(image_full, (5, 5), 0)
blurred_half = cv2.GaussianBlur(image_half, (5, 5), 0)
blurred_quart = cv2.GaussianBlur(image_quart, (5, 5), 0)
blurred_eighth = cv2.GaussianBlur(image_eighth, (5, 5), 0)

# Edge detection
t1 = 30
t2 = 100
Im_1 = cv2.Canny(blurred_full, threshold1=t1, threshold2=t2)
Im_2 = cv2.Canny(blurred_half, threshold1=t1, threshold2=t2)
Im_3 = cv2.Canny(blurred_quart, threshold1=t1, threshold2=t2)
Im_4 = cv2.Canny(blurred_eighth, threshold1=t1, threshold2=t2)

# Load ground_truth_images
Im_1_gt = cv2.imread('./images/DALLE-camel-truth.png', cv2.IMREAD_GRAYSCALE)
Im_2_gt = cv2.imread('./images/DALLE-camel-truth-half.png', cv2.IMREAD_GRAYSCALE)
Im_3_gt = cv2.imread('./images/DALLE-camel-truth-quart.png', cv2.IMREAD_GRAYSCALE)
Im_4_gt = cv2.imread('./images/DALLE-camel-truth-eighth.png', cv2.IMREAD_GRAYSCALE)

# Create a DataFrame to store the evaluation metrics
data = {
    "Image": ["1024p", "512p", "256p", "128p"],
    "Precision": [],
    "Recall": [],
    "F1-score": []}


# Calculate evaluation metrics for each result and add to the DataFrame
average = 'weighted'
for edges, truth, label in zip([Im_1, Im_2, Im_3, Im_4], [Im_1_gt, Im_2_gt, Im_3_gt, Im_4_gt], data["Image"]):
    edges_binary = (edges > 0).astype(np.uint8) * 255
    truth_binary = (truth > 0).astype(np.uint8) * 255
    precision = precision_score(truth_binary, edges_binary, average=average)
    recall = recall_score(truth_binary, edges_binary, average=average)
    f1 = f1_score(truth_binary, edges_binary, average=average)

    data["Precision"].append(precision)
    data["Recall"].append(recall)
    data["F1-score"].append(f1)

# Create a pandas DataFrame
df = pd.DataFrame(data)

print(df)

plt.figure(figsize=(30, 60))
fontsize = 60

plt.subplot(421)
plt.imshow(Im_1, cmap='gray')
plt.title('Reference 1024p image', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(422)
plt.imshow(Im_1_gt, cmap='gray')
plt.title('Reference 1024p ground truth', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(423)
plt.imshow(Im_2, cmap='gray')
plt.title('512p image', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(424)
plt.imshow(Im_2_gt, cmap='gray')
plt.title('512p ground truth', fontsize=fontsize)
plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.subplot(425)
plt.imshow(Im_3, cmap='gray')
plt.title('256p image', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(426)
plt.imshow(Im_3_gt, cmap='gray')
plt.title('256p ground truth', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(427)
plt.imshow(Im_4, cmap='gray')
plt.title('128p image', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(428)
plt.imshow(Im_4_gt, cmap='gray')
plt.title('128p ground truth', fontsize=fontsize)
plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.show()

#%%