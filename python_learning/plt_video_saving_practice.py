"""
This file is primarily used to practice saving videos of matplotlib plots.

We will create a function that creates a simple plot given an input... we can then loop over this over time to create
something we can take a video of.
"""

# Good source: https://www.geeksforgeeks.org/matplotlib-animation-funcanimation-class-in-python/

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def create_plot(x, y):
    """
    This function creates a simple plot given an input x and y.
    """
    # Create a figure
    fig = plt.figure()
    # Create an axis
    ax = fig.add_subplot(111)
    # Plot the data
    ax.plot(x, y)
    # Return the figure
    return fig


def update(i):
    """
    This function updates the plot given an input i.
    """
    # Clear the axis
    # ax.clear()
    # Plot the data
    ax.plot(x, y)
    # Set the title
    ax.set_title(f"i = {i}")
    # Return the axis
    return ax


# Create some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the figure
fig = create_plot(x, y)
# Create the axis
ax = fig.add_subplot(111)

# matplotlib.animation.FuncAnimation class is used to make animation by repeatedly calling the same function (ie, func).
ani = FuncAnimation(fig, update, frames=np.arange(0, 10, 0.1), interval=100)  # Frames is passed to the update function

# Save the animation
ani.save("test.gif")
