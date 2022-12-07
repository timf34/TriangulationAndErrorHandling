import cv2
import numpy as np
from matplotlib import pyplot as plt

# Create a VideoWriter object to pass our pyplot figure to
writer = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, (640, 480))

# Create a pyplot figure
fig = plt.figure()

# Loop through the frames
for i in range(100):
    # Create a random image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Plot the image
    plt.imshow(img)

    # TODO: note that this is super ridiculously slow.
    # Write the frame to the VideoWriter object
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    writer.write(img)



# Release the VideoWriter object
writer.release()

# Close the pyplot figure
plt.close(fig)



