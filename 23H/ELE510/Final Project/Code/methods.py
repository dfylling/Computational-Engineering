#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import filters
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the original image
image = cv2.imread('./images/DALLE-camel.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian smoothing
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Edge detection using different methods
edges_canny = cv2.Canny(blurred, threshold1=30, threshold2=100)
edges_sobel = filters.sobel(blurred)
edges_prewitt = filters.prewitt(blurred)
edges_log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)  # Laplacian of Gaussian

# Normalize the LoG edges
log_1 = edges_log - np.min(edges_log)
log_norm =  (log_1 / np.max(log_1))

# Threshold edges from Sobel, Prewitt, and Laplacian of Gaussian 
thresh = 0.07  
edges_sobel_binary = (edges_sobel > thresh).astype(np.uint8) * 255
edges_prewitt_binary = (edges_prewitt > thresh).astype(np.uint8) * 255
thresh = 0.61  
edges_log_binary = (log_norm > thresh).astype(np.uint8) * 255

# Load ground truth image
ground_truth = cv2.imread('./images/DALLE-camel-truth.png', cv2.IMREAD_GRAYSCALE)
ground_truth_dilated = cv2.imread('./images/DALLE-camel-truth-dilated.png', cv2.IMREAD_GRAYSCALE)
ground_truth_binary = (ground_truth > 0).astype(np.uint8) * 255

# Create a DataFrame to store the evaluation metrics
data = {
    "Method": ["Canny", "Sobel", "Prewitt", "LoG"],
    "Precision": [],
    "Recall": [],
    "F1-score": []
}

# Calculate evaluation metrics for each method and add to the DataFrame
average = 'weighted'
for edges_binary, method in zip([edges_canny, edges_sobel_binary, edges_prewitt_binary, edges_log_binary], data["Method"]):
    precision = precision_score(ground_truth_binary, edges_binary, average=average)
    recall = recall_score(ground_truth_binary, edges_binary, average=average)
    f1 = f1_score(ground_truth_binary, edges_binary, average=average)

    data["Precision"].append(precision)
    data["Recall"].append(recall)
    data["F1-score"].append(f1)

# Create a pandas DataFrame
df = pd.DataFrame(data)

print(df)

plt.figure(figsize=(30, 45))
fontsize = 60

plt.subplot(321)
plt.imshow(ground_truth, cmap='gray')
plt.title('Ground Truth Image', fontsize=fontsize)
plt.axis('off')

plt.subplot(322)
plt.imshow(ground_truth_dilated, cmap='gray')
plt.title('Dilated Ground Truth Image', fontsize=fontsize)
plt.axis('off')

plt.subplot(323)
plt.imshow(edges_canny, cmap='gray')
plt.title('Canny Edge Detection', fontsize=fontsize)
plt.axis('off')

plt.subplot(324)
plt.imshow(edges_sobel_binary, cmap='gray')
plt.title('Sobel Edge Detection', fontsize=fontsize)
plt.axis('off')

plt.subplot(325)
plt.imshow(edges_prewitt_binary, cmap='gray')
plt.title('Prewitt Edge Detection', fontsize=fontsize)
plt.axis('off')

plt.subplot(326)
plt.imshow(edges_log_binary, cmap='gray')
plt.title('LoG Edge Detection', fontsize=fontsize)
plt.axis('off')

plt.tight_layout()
plt.show()


#%%