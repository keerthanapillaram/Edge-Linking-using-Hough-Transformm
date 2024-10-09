# Edge-Linking-using-Hough-Transformm
## Aim:
To write a Python program to detect the lines using Hough Transform.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:

Import the necessary libraries: cv2 for OpenCV operations, numpy for mathematical operations, and matplotlib to display the images.
### Step2:

Load the image with cv2.imread().
### Step3:

Convert the color image to grayscale using cv2.cvtColor().
### Step4:

 Detect the edges using cv2.Canny(). The function parameters include: 50: Lower threshold for edge detection. 150: Upper threshold for edge detection. apertureSize=3: Kernel size for Sobel operator used in edge detection.
### Step5:

Use HoughLinesP() for the probabilistic Hough Line Transform. It returns line segments, and for each line segment, it provides the coordinates of the endpoints (x1, y1, x2, y2). The detected lines are drawn on a copy of the original image.
## Program:
```python
Developed by: P Keerthana
Reg.No: 212223240069

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('HappyFish.jpg')  # Replace with your image path

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Displaying results using Matplotlib
plt.figure(figsize=(12, 12))

# Input Image and Grayscale Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
```
### i)Input image and grayscale image
![image](https://github.com/user-attachments/assets/e868704c-2046-4085-b1d1-da0e64db5a5f)
<br>
```python
edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

# Canny Edge Detection Output
plt.subplot(2, 2, 3)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detector Output')
plt.axis('off')
```
### ii)Canny Edge detector output
![image](https://github.com/user-attachments/assets/6fc4d701-fa7c-4761-8f6b-f8e00b4d4dd1)
<br>
```python
# Detect lines using the probabilistic Hough transform
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Draw the lines on the original image
output_image = image.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Hough Transform Result
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('Hough Transform - Line Detection')
plt.axis('off')

# Display all results
plt.show()

```
### iii)Display the result of Hough transform
![image](https://github.com/user-attachments/assets/6f3cffd0-0c7d-44be-b778-66f8760de737)
<br>
## Result:
Successfully converted the given image to Canny edge detector and Hough transform method and displayed using python.


