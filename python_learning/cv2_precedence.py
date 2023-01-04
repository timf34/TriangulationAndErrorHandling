"""
I want to be able to draw a number using cv2.putText, and have it appear on top of a circle drawn with cv2.circle
"""
import cv2
import numpy as np

# Create a black image
img = np.zeros((512,512,3), np.uint8)


# Put text in the centre - just a single digit
cv2.putText(img, "1", (256, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Draw a circle in the centre
cv2.circle(img, (256, 256), 20, (0, 255, 0), -3)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# The text is not visible on top of the circle
# Lets try to fix that!

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a circle in the centre
cv2.circle(img, (256, 256), 20, (0, 255, 0), -3)

# Put text in the centre - just a single digit
cv2.putText(img, "1", (256, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# The text is now visible on top of the circle
# However, I want to be able to call cv2.putText before cv2.circle, and still have the text appear on top of the circle
# Is this possible?

# No, it doesn't seem so.



