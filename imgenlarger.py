import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

""" Note : If your image is .png convert it to .jpg , otherwise it won't work """

# Load the image using Matplotlib
image_file = "wojak.jpg"
image = plt.imread(image_file)

# Check if the image has four channels
if image.shape[2] == 4:
    rgba_image = Image.open(image_file)

    # Convert RGBA to RGB (discard alpha channel)
    rgb_image = rgba_image.convert("RGB")

    # Save the RGB image
    rgb_image.save(image_file)
    image = plt.imread(image_file)
else:
    pass

# Create a numpy array from our image
matrix = np.array(image)
matrix = matrix.astype(np.uint8)

# Convert RGB values to hexadecimal values
hex_array = np.array([['#{:02x}{:02x}{:02x}'.format(*pixel) for pixel in row] for row in matrix])

# Shape Change
matrix = hex_array.T

# Print the resulting matrix
print(matrix)
print("Shape of the matrix:", matrix.shape)

# Display the image using Matplotlib
plt.imshow(image)
plt.axis('off')
plt.show()


print("=========================       2x       ==============================")

# Convert our matrix numpy array to a python list
matrix = matrix.tolist()

# Doubling the pixel rows
for i in matrix:
    oddpattern = -1
    evenpattern = 0
    for j in range(0, len(i)):
        oddpattern += 2
        i.insert(oddpattern, i[evenpattern])
        evenpattern += 2

# Doubling pixel columns
evenpattern = 0
oddpattern = -1
for i in range(0, len(matrix)):
    oddpattern += 2
    matrix.insert(oddpattern, matrix[evenpattern])
    evenpattern += 2

# Reconvert python list to numpy array
matrix = np.array(matrix)

print(matrix)
print("Shape of the matrix 2x:", matrix.shape)
matrix = matrix.T


#=========================       Filter       ==============================

# Convert our hex matrix np array to RGB values
rgb_array = np.array([[tuple(int(hex_val[i:i+2], 16) for i in (1, 3, 5)) for hex_val in row] for row in matrix])
rgb_array = rgb_array.astype(np.uint8)

# Convert RGB array to BGR (OpenCV uses BGR format)
bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

# If you want to use Gaussian blur
# smoothed_bgr_array = cv2.GaussianBlur(bgr_array, (3, 3), 0)

# Apply Median blur
smoothed_bgr_array = cv2.medianBlur(bgr_array, 3)

# Convert back to RGB
smoothed_rgb_array = cv2.cvtColor(smoothed_bgr_array, cv2.COLOR_BGR2RGB)

# Display the smoothed image
plt.imshow(smoothed_rgb_array)

# Save the smoothed image
plt.imsave("generatedimage/enlargedimage.jpg", smoothed_rgb_array)
plt.axis('off')
plt.show()
