import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def create_plot_with_multiple_axes(x, y):
    """
    This function creates a simple plot given an input x and y.
    """
    # Create a figure
    fig = plt.figure()
    # Create an axis
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # Plot the data
    ax1.plot(x, y)
    ax2.plot(x, y)
    # Return the figure
    return fig, ax1, ax2


def run():
    # Create some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the figure
    fig, ax1, ax2 = create_plot_with_multiple_axes(x, y)

    def update(i):
        """
        This function updates the plot given an input i.
        """
        # Clear the axis
        # ax.clear()
        # Plot the data
        ax1.plot(x, y)
        ax2.plot(x, y)
        # Set the title
        ax1.set_title(f"i = {i}")
        ax2.set_title(f"i = {i}")
        # Return the axis
        return ax1, ax2

    # matplotlib.animation.FuncAnimation class is used to make animation by repeatedly calling the same function (ie, func).
    ani = FuncAnimation(fig, update, frames=np.arange(0, 10, 0.1), interval=100)  # Frames is passed to the update function

    # Save the animation
    ani.save("testing_yo.gif")


def update_subplot(ax, data):
    pass


def main():
    run()


if __name__ == "__main__":
    main()
