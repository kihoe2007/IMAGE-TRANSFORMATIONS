# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
<br>Load the image

### Step2:
<br> Image Translation

### Step3:
<br>Image Scaling

### Step4:
<br>Image Shearing

### Step5:
<br>Image Reflection

## Program:
```python
Developed By :Kishore S M
Reg no:212224230131
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display image using Matplotlib
def display_image(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper color display
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load an image
image = cv2.imread('tree.jpg')
display_image(image, 'Original Image')

# i) Image Translation
def translate(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return translated

translated_image = translate(image, 100, 50)
display_image(translated_image, 'Translated Image')

# ii) Image Scaling
def scale(img, scale_x, scale_y):
    scaled = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return scaled

scaled_image = scale(image, 1.5, 1.5)
display_image(scaled_image, 'Scaled Image')

# iii) Image Shearing
def shear(img, shear_factor):
    rows, cols, _ = img.shape
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(img, M, (cols, rows))
    return sheared

sheared_image = shear(image, 0.5)
display_image(sheared_image, 'Sheared Image')

# iv) Image Reflection
def reflect(img):
    reflected = cv2.flip(img, 1)  # 1 for horizontal flip
    return reflected

reflected_image = reflect(image)
display_image(reflected_image, 'Reflected Image')

# v) Image Rotation
def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

rotated_image = rotate(image, 45)
display_image(rotated_image, 'Rotated Image')

# vi) Image Cropping
def crop(img, start_row, start_col, end_row, end_col):
    cropped = img[start_row:end_row, start_col:end_col]
    return cropped

cropped_image = crop(image, 50, 50, 200, 200)
display_image(cropped_image, 'Cropped Image')

```
## Output:
### i)Image Translation
<br>
<img width="936" height="691" alt="image" src="https://github.com/user-attachments/assets/da0e7f14-3748-40f3-80b6-c3add3c008ef" />

<br>
<br>
<br>

### ii) Image Scaling
<br>
<br><img width="954" height="546" alt="image" src="https://github.com/user-attachments/assets/889b61b4-5173-4caa-9c61-8033785333e4" />

<br>
<br>


### iii)Image shearing
<br>
<br>
<br><img width="1022" height="669" alt="image" src="https://github.com/user-attachments/assets/c91f9449-9a11-419f-9ed0-1c2c0cea0e11" />

<br>


### iv)Image Reflection
<br>
<br><img width="1026" height="698" alt="image" src="https://github.com/user-attachments/assets/3130a497-39c9-4f5a-b0ac-953cf1b47871" />

<br>
<br>



### v)Image Rotation
<br>
<br><img width="1024" height="644" alt="image" src="https://github.com/user-attachments/assets/bfbc3139-86ae-423b-af29-4ac06d3f4ca7" />

<br>
<br>



### vi)Image Cropping
<br>
<br><img width="891" height="635" alt="image" src="https://github.com/user-attachments/assets/3e306f9f-3816-431f-9275-9e7e0f6ff0bc" />

<br>
<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
