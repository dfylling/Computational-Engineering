#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt  

# Load the image
image = cv2.imread('./images/DALLE-camel.png', cv2.IMREAD_GRAYSCALE)

# Compute gradient magnitude and phase
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
G_mag = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
G_phase = np.arctan2(gradient_y, gradient_x)

# Non-maximum suppression
def nonMaxSuppression(G_mag, G_phase):
    G_localmax = np.zeros((G_mag.shape))
    X, Y = G_mag.shape

    # For each pixel, adjust the phase to ensure that -pi/8 <= theta < 7*pi/8
    for x in range(X-2):
        x=x+1
        for y in range(Y-2):
            y=y+1
            theta = G_phase[x,y]
            if theta >= np.pi*0.875:
                theta = theta - np.pi
            if theta < -np.pi*0.125:
                theta = theta + np.pi
            if -np.pi*0.125 <= theta < np.pi*0.125:
                n1 = G_mag[x-1, y]
                n2 = G_mag[x+1, y]
            if np.pi*0.125 <= theta < np.pi*0.375:
                n1 = G_mag[x-1, y-1]
                n2 = G_mag[x+1, y+1]
            if np.pi*0.375 <= theta < np.pi*0.625:
                n1 = G_mag[x, y-1]
                n2 = G_mag[x, y+1]
            if np.pi*0.625 <= theta < np.pi*0.875:
                n1 = G_mag[x, y-1]
                n2 = G_mag[x, y+1]               
            if (G_mag[x,y] >= n1) & (G_mag[x,y] >= n2):
                G_localmax[x,y] = G_mag[x,y]
    
    return G_localmax
G_localmax = nonMaxSuppression(G_mag, G_phase)


# Edge tracking by hysteresis
I_edges = cv2.Canny(image, threshold1=30, threshold2=100)

# Create a figure with subplots to display the intermediate results
plt.figure(figsize=(30, 30))
fontsize = 60

plt.subplot(221)
plt.imshow(G_mag, cmap='gray')
plt.title('Magnitude image', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(222)
plt.imshow(G_phase, cmap='gray')
plt.title('Phase image', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(223)
plt.imshow(G_localmax, cmap='gray')
plt.title('After non maximum suppression', fontsize=fontsize)
plt.xticks([]), plt.yticks([])

plt.subplot(224)
plt.imshow(I_edges, cmap='gray')
plt.title('Threshold image', fontsize=fontsize)
plt.xticks([]), plt.yticks([])
plt.tight_layout()

plt.show()

#%%