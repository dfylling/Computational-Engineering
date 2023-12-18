#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from skimage.util import random_noise


# Load the image
image = cv2.imread('./images/DALLE-camel.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian smoothing
blurred_image_5 = cv2.GaussianBlur(image, (5, 5), 0)

# Apply degrees of Gaussian white noise
def add_gauss_noise(image, mode='gaussian',mean=0, var=0.1, amount=0.05):
    if mode == 'gaussian':
        noisy_image = random_noise(image,  mode=mode, mean=mean, var=var)
    if mode == 's&p':
        noisy_image = random_noise(image,  mode=mode, amount = amount)
    noisy_image = np.clip(noisy_image, 0, 255) * 255
    return noisy_image.astype(np.uint8)

noisy_image_1 = add_gauss_noise(blurred_image_5, mode='gaussian',mean=0, var=0.0001, amount=0.05)
noisy_image_2 = add_gauss_noise(blurred_image_5, mode='gaussian',mean=0, var=0.001, amount=0.05)
noisy_image_3 = add_gauss_noise(blurred_image_5, mode='gaussian',mean=0, var=0.01, amount=0.05)

# Edge detection
t1 = 30
t2 = 100
Im_1 = cv2.Canny(blurred_image_5, threshold1=t1, threshold2=t2)
Im_2 = cv2.Canny(noisy_image_1, threshold1=t1, threshold2=t2)
Im_3 = cv2.Canny(noisy_image_2, threshold1=t1, threshold2=t2)
Im_4 = cv2.Canny(noisy_image_3, threshold1=t1, threshold2=t2)

# Load ground_truth_image
ground_truth_image = cv2.imread('./images/DALLE-camel-truth.png', cv2.IMREAD_GRAYSCALE)
ground_truth_binary = (ground_truth_image > 0).astype(np.uint8) * 255

# Create a DataFrame to store the evaluation metrics
data = {
    "Image": ["Recerence 5x5", "Some noise", "Medium noise", "Lots of noise"],
    "Precision": [],
    "Recall": [],
    "F1-score": []}


# Calculate evaluation metrics for each result and add to the DataFrame
average = 'weighted'
for edges, label in zip([Im_1, Im_2, Im_3, Im_4], data["Image"]):
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

plt.figure(figsize=(30, 60))
fontsize = 40

plt.subplot(421)
plt.imshow(blurred_image_5, cmap='gray')
plt.title('Reference 5x5 blurred', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(422)
plt.imshow(Im_1, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(423)
plt.imshow(noisy_image_1, cmap='gray')
plt.title('Some noise', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(424)
plt.imshow(Im_2, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.subplot(425)
plt.imshow(noisy_image_2, cmap='gray')
plt.title('Medium noise', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(426)
plt.imshow(Im_3, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(427)
plt.imshow(noisy_image_3, cmap='gray')
plt.title('Lots of noise', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(428)
plt.imshow(Im_4, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.show()
#%%