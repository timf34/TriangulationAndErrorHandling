"""
    This file will be for creating a plt plot, and then pressing a key to close it and move on
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Create a blank image for plotting
image = np.zeros((100, 100, 3), dtype=np.uint8)

# Plot the image
plt.imshow(image)

# Show the plot
plt.show(block=False)  # Note: block=False is needed to allow us to close the plot via a key press

# Wait for user to press a key before showing next plot
# plt.waitforbuttonpress(0)  # This isn't working for some reason, it stays open. We should use cv2.waitKey(0) instead

# Wait for user to click their mouse before showing next plot; we should use this instead of waitforbuttonpress(0)
plt.waitforbuttonpress()

# Show the plot again, but change the image
image[:, :, 1] = 255
plt.imshow(image)
plt.show()

# Wait for user to press a key before closing and ending the script
plt.waitforbuttonpress(0)
plt.close()

print("Script finished")
