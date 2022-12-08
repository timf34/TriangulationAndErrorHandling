import cv2
import numpy as np

# Create a black image
img1 = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
img1 = cv2.line(img1,(0,0),(511,511),(255,0,0),5)

# Create a white image
img = np.ones((512,512,3), np.uint8)
# Multiply by 255 to get white
img = img * 255

# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

# Stack the images together
stacked_image = np.hstack((img1, img))

# Show images
cv2.imshow("Stacked image", stacked_image)
cv2.waitKey(0)

# Get the image size
height, width, layers = stacked_image.shape
print(height, width, layers)


# Create a cv2 VideoWriter object
out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, (width, height))

# Create a for loop to write the frames to the video
for i in range(10):
    out.write(stacked_image)

# Release the VideoWriter object
out.release()

